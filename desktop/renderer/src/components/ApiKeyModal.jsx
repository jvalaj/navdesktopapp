import React, { useState } from "react";

export default function ApiKeyModal({ onSaved }) {
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [skipped, setSkipped] = useState(false);

  const save = async () => {
    if (!key.trim()) {
      setError("Please enter a key");
      return;
    }
    if (!window.navai?.apiKeySet) {
      setError("Preload bridge not available. Please restart the app.");
      return;
    }
    const result = await window.navai.apiKeySet(key.trim());
    if (result?.ok) {
      await window.navai?.pythonRestart();
      onSaved?.();
    } else {
      setError(result?.error || "Could not save key");
    }
  };

  if (skipped) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h3>Enter your Anthropic API key</h3>
        <p style={{ color: "#93B48B", fontSize: 13 }}>
          Your key is stored securely in the OS keychain. It is never shipped or hardcoded.
        </p>
        <input
          type="password"
          placeholder="sk-ant-..."
          value={key}
          onChange={(e) => setKey(e.target.value)}
          style={{ width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid rgba(132,145,163,0.35)" }}
        />
        {error && <div style={{ color: "#c96b6b", fontSize: 12, marginTop: 6 }}>{error}</div>}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
          <button className="btn-outline" onClick={() => setSkipped(true)}>Skip for now</button>
          <button className="btn-primary" onClick={save}>Save Key</button>
        </div>
      </div>
    </div>
  );
}
