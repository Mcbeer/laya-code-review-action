from laya_review.diff import Chunk, chunk_diff, render_state


def count_words(text: str) -> int:
    return len(text.split())


DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,4 @@ def main():
     setup()
+    run()
     teardown()
@@ -20,2 +21,2 @@ def helper():
-    old()
     keep()
diff --git a/src/gone.py b/src/gone.py
deleted file mode 100644
--- a/src/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-print("bye")
-print("bye")
diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
diff --git a/docs/new file.md b/docs/new file.md
new file mode 100644
--- /dev/null
+++ b/docs/new file.md
@@ -0,0 +1 @@
+hello
"""


def test_keeps_only_hunks_that_add_lines_in_live_text_files():
    chunks = chunk_diff(DIFF, count_words, budget=1000)
    assert chunks == [
        Chunk("src/app.py", "@@ -1,3 +1,4 @@ def main():", "     setup()\n+    run()\n     teardown()"),
        Chunk("docs/new file.md", "@@ -0,0 +1 @@", "+hello"),
    ]


def test_splits_large_hunk_within_budget_and_repeats_header():
    body = "\n".join(f"+line{i}" for i in range(10))
    diff = f"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -0,0 +1,10 @@\n{body}\n"
    budget = 12

    chunks = chunk_diff(diff, count_words, budget)

    assert len(chunks) > 1
    assert all(c.hunk == "@@ -0,0 +1,10 @@" for c in chunks)
    assert "\n".join(c.body for c in chunks) == body
    assert all(count_words(c.as_state()) <= budget for c in chunks)


def test_split_drops_pieces_without_added_lines():
    body = "\n".join([" ctx1", " ctx2", " ctx3", "+new"])
    diff = f"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,4 @@\n{body}\n"

    chunks = chunk_diff(diff, count_words, budget=count_words(render_state("a.py", "@@ -1,3 +1,4 @@", "")) + 2)

    assert all("+" in c.body for c in chunks)
    assert chunks[-1].body.endswith("+new")


def test_chunking_is_deterministic():
    assert chunk_diff(DIFF, count_words, 5) == chunk_diff(DIFF, count_words, 5)
