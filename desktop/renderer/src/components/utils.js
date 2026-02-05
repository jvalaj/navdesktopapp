export const uid = () => Math.random().toString(36).slice(2);

export const nowLabel = () => {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

/** Resolve screenshot path to full path using app storage screenshots dir when path is relative. */
export function resolveScreenshotPath(path, screenshotsDir) {
  if (!path) return "";
  const isAbsolute = path.startsWith("/") || (path.length >= 2 && path[1] === ":");
  if (isAbsolute) return path;
  if (screenshotsDir) {
    const basename = path.replace(/^.*[/\\]/, "");
    return `${screenshotsDir}/${basename}`;
  }
  return path;
}
