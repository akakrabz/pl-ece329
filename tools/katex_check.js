// Compile every expression with KaTeX; print the failures as JSON. Usage: node katex_check.js path/to/katex.min.js < items.json
const fs = require("fs");
const katex = require(process.argv[2]);
const items = JSON.parse(fs.readFileSync(0, "utf8"));
const bad = [];
for (const it of items) {
  try {
    katex.renderToString(it.e, { displayMode: !!it.d, throwOnError: true, strict: "ignore" });
  } catch (err) {
    bad.push({ q: it.q, e: it.e, err: String(err.message || err) });
  }
}
process.stdout.write(JSON.stringify(bad));
