#!/usr/bin/env node
const fs = require("node:fs/promises");
const path = require("node:path");

let chromium;
try {
  ({ chromium } = require("playwright"));
} catch (error) {
  console.error("Cannot load Playwright. Run through scripts/capture_ui_current.sh.");
  console.error(error.message);
  process.exit(1);
}

const baseUrl = process.env.UI_CAPTURE_URL || "http://127.0.0.1:8899";
const outDir = path.resolve(process.env.UI_CAPTURE_OUT || "docs/scrollytelling_audit/_ui_current");
const [width, height] = (process.env.UI_CAPTURE_VIEWPORT || "1440,900")
  .split(",")
  .map((value) => Number.parseInt(value, 10));
const waitMs = Number.parseInt(process.env.UI_CAPTURE_WAIT_MS || "900", 10);

const shots = [
  ["s-hero", "hero.png"],
  ["s-intro", "intro.png"],
  ["s-cover", "cover.png"],
  ["s-article", "article.png"],
  ["s-stages", "stages.png"],
  ["s-bridge", "bridge.png"],
  ["s-state", "state.png"],
  ["s-network", "network.png"],
  ["s-quote", "quote.png"],
  ["s-electric", "electric.png"],
  ["s-water", "water.png"],
  ["s-stats", "stats.png"],
  ["s-floor", "floor.png"],
  ["s-debate", "debate.png"],
  ["s-final", "final.png"],
];

async function main() {
  await fs.mkdir(outDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width, height },
    deviceScaleFactor: 1,
  });

  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${baseUrl}/?static`, { waitUntil: "networkidle" });
  await page.waitForSelector(".landing section");
  await page.addStyleTag({
    content: `
      html { scroll-behavior: auto !important; }
      *, *::before, *::after {
        animation-delay: 0s !important;
        animation-duration: 0s !important;
        transition-delay: 0s !important;
        transition-duration: 0s !important;
      }
      .chat-panel { display: none !important; }
    `,
  });

  for (const [id, name] of shots) {
    const output = path.join(outDir, name);
    console.log(`capture ${id} -> ${path.relative(process.cwd(), output)}`);
    await page.evaluate((targetId) => {
      const target = document.getElementById(targetId);
      if (!target) throw new Error(`Missing section: ${targetId}`);

      document.querySelectorAll(".landing section").forEach((section) => {
        section.style.display = section.id === targetId ? "" : "none";
      });

      const nav = document.getElementById("chapterNav");
      if (nav) nav.style.position = "static";

      window.scrollTo(0, 0);
    }, id);
    await page.waitForSelector(`#${id}`, { state: "visible" });
    await page.waitForTimeout(waitMs);
    await page.screenshot({ path: output, fullPage: false });
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
