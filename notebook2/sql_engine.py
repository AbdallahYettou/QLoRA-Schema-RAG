"""Chunk 4: lossless comment stripping and bounded, read-only SQLite scoring."""
from contextlib import closing
import logging
from pathlib import Path
import re
import sqlite3
import time

import pandas as pd
import sqlglot
from sqlglot import exp


def strip_comments(sql):
    """Remove comments outside quoted literals/identifiers, preserving SQL spelling.

    Re-rendering an AST can change result column labels (COUNT(*) vs count(*)),
    so keep the original query for strict Pandas equality with Notebook 1.
    """
    out, i, quote = [], 0, None
    while i < len(sql):
        char = sql[i]
        if quote:
            out.append(char)
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote and quote != "]":
                    out.append(sql[i + 1])
                    i += 1
                else:
                    quote = None
        elif char in "'\"`[":
            quote = "]" if char == "[" else char
            out.append(char)
        elif sql[i:i+2] == "--":
            end = sql.find("\n", i + 2)
            i = len(sql) if end == -1 else end
            out.append("\n")
            continue
        elif sql[i:i+2] == "/*":
            end = sql.find("*/", i + 2)
            if end == -1:
                raise ValueError("Unterminated SQL comment")
            i = end + 2
            out.append(" ")
            continue
        else:
            out.append(char)
        i += 1
    if quote:
        raise ValueError("Unterminated SQL quote")
    return "".join(out).strip()


def parse_readonly(sql):
    statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("Expected exactly one SELECT/CTE query")
    # Block modifying CTEs as well as top-level mutations.
    if any(isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop))
           for node in statements[0].walk()):
        raise ValueError("Modifying statements are forbidden")
    return statements[0]


def sanitize_sql(answer):
    blocks = re.findall(r"```(?:sql|sqlite)?\s*\n?(.*?)```", answer, re.I | re.S)
    if len(blocks) > 1 or ("```" in answer and not blocks):
        raise ValueError("Expected one complete SQL code block")
    sql = strip_comments(blocks[0] if blocks else answer)
    parse_readonly(sql)
    return sql


def execute_dataframe(path, sql, timeout=5.0, max_rows=100000):
    if timeout <= 0 or max_rows <= 0:
        raise ValueError("Positive timeout and row limit required")
    parse_readonly(sql)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
               sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    def authorize(action, arg1, arg2, db, source):
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
    start = time.monotonic()
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.set_authorizer(authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 1000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError("Result row limit exceeded")
        return pd.DataFrame.from_records(rows, columns=[column[0] for column in cursor.description])


def score_prediction(path, answer, gold):
    result = {"sql": "", "syntax_valid": False, "execution_valid": False,
        "execution_match": False, "error_type": None, "error": None}
    try:
        result["sql"] = sanitize_sql(answer)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        return {**result, "error_type": "invalid_sql", "error": str(exc)}
    result["syntax_valid"] = True
    try:
        predicted = execute_dataframe(path, result["sql"])
        result.update(execution_valid=True, execution_match=gold is not None and predicted.equals(gold),
                      result_shape=list(predicted.shape))
    except sqlite3.Error as exc:
        syntax_error = any(marker in str(exc).lower() for marker in
                           ("syntax error", "incomplete input", "unrecognized token"))
        result.update(syntax_valid=not syntax_error,
            error_type="sqlite_syntax" if syntax_error else "sqlite_execution", error=str(exc))
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        result.update(error_type="resource_or_query_limit", error=str(exc))
    return result


def verify_sql_engine():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.sqlite"
        with sqlite3.connect(path) as conn:
            conn.executescript("CREATE TABLE item(id INTEGER PRIMARY KEY, name TEXT);"
                               "INSERT INTO item VALUES (1,'a'),(2,'b');")
        gold = execute_dataframe(path, "SELECT count(*) FROM item")
        assert score_prediction(path, "```sql\nSELECT count(*) FROM item; -- done\n```", gold)["execution_match"]
        assert not score_prediction(path, "SELECT count(*) AS n FROM item", gold)["execution_match"]
        assert sanitize_sql("SELECT '--literal/*x*/' AS x -- note") == "SELECT '--literal/*x*/' AS x"
        assert not score_prediction(path, "SELECT FROM", gold)["syntax_valid"]
        missing = score_prediction(path, "SELECT * FROM nonexistent", gold)
        assert missing["syntax_valid"] and not missing["execution_valid"]
        for sql in ("DELETE FROM item", "SELECT 1; DROP TABLE item", "ATTACH DATABASE 'x' AS x",
                    "SELECT load_extension('x')", "SELECT 1 /* unclosed"):
            assert not score_prediction(path, sql, gold)["execution_valid"]
        assert not score_prediction(path, "```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```", gold)["syntax_valid"]
        try:
            execute_dataframe(path, "SELECT * FROM item", max_rows=1)
            raise AssertionError("Row cap not enforced")
        except ValueError:
            pass
        try:
            execute_dataframe(path, "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x", timeout=.01)
            raise AssertionError("Timeout not enforced")
        except sqlite3.OperationalError as exc:
            assert "interrupted" in str(exc)
        assert execute_dataframe(path, "SELECT count(*) FROM item").equals(gold)
    logging.info("CHUNK 4 PASS | sanitation, strict equality, syntax/errors, read-only, timeout, row cap")
