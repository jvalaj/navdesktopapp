export const uid = () => Math.random().toString(36).slice(2);

export const nowLabel = () => {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};
