import React, { useState } from "react";

export default function SettingsModal({ settings, onClose, onSave, apiKeyStatus, onAddApiKey }) {
  const [form, setForm] = useState(settings);

  const update = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Settings</h3>
        
        {/* API Key Section */}
        <div style={{ 
          marginBottom: 24, 
          padding: 16, 
          background: apiKeyStatus?.connected ? "rgba(152, 206, 0, 0.1)" : "rgba(201, 107, 107, 0.1)",
          borderRadius: 12,
          border: `1px solid ${apiKeyStatus?.connected ? "rgba(152, 206, 0, 0.3)" : "rgba(201, 107, 107, 0.3)"}`
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                API Key
              </div>
              <div style={{ fontSize: 12, color: "var(--muted)" }}>
                {apiKeyStatus?.connected ? "API key is configured" : "API key not added"}
              </div>
            </div>
            {!apiKeyStatus?.connected && (
              <button 
                className="btn-primary" 
                onClick={() => {
                  onClose();
                  onAddApiKey?.();
                }}
                style={{ padding: "8px 16px", fontSize: 13 }}
              >
                Add API Key
              </button>
            )}
          </div>
        </div>

        <div className="settings-grid">
          <label>
            Model
            <select value={form.model} onChange={(e) => update("model", e.target.value)}>
              <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5</option>
              <option value="claude-opus-4-5-20250929">Claude Opus 4.5</option>
            </select>
          </label>
          <label>
            Screenshot Frequency (seconds)
            <input
              type="number"
              min={1}
              max={10}
              value={form.screenshotFrequency}
              onChange={(e) => update("screenshotFrequency", Number(e.target.value))}
            />
          </label>
          <label>
            Save Screenshots
            <select value={form.saveScreenshots ? "on" : "off"} onChange={(e) => update("saveScreenshots", e.target.value === "on")}>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
          <label>
            Allow Sending Screenshots
            <select value={form.allowSendScreenshots ? "on" : "off"} onChange={(e) => update("allowSendScreenshots", e.target.value === "on")}>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
          <label>
            Dry Run Mode
            <select value={form.dryRun ? "on" : "off"} onChange={(e) => update("dryRun", e.target.value === "on")}>
              <option value="off">Execute</option>
              <option value="on">Plan Only</option>
            </select>
          </label>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
          <button className="btn-outline" onClick={onClose}>Cancel</button>
          <button className="btn-primary w-1/2" onClick={() => onSave(form)}>Save</button>
        </div>
      </div>
    </div>
  );
}
