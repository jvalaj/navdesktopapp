import React, { useState } from "react";

export default function SettingsModal({ settings, onClose, onSave, apiKeyStatus, onAddApiKey }) {
  const [form, setForm] = useState(settings);
  const provider = form.modelProvider || "anthropic";
  const providerKeyConnected = Boolean(apiKeyStatus?.providers?.[provider]);
  const providerLabel = provider === "anthropic"
    ? "Anthropic Claude"
    : provider === "openai"
      ? "OpenAI ChatGPT"
      : provider === "gemini"
        ? "Google Gemini"
        : "Z.ai GLM";

  const update = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));
  const updateNumber = (key, value, min, max) => {
    const numeric = Number.parseInt(value, 10);
    if (Number.isNaN(numeric)) return;
    const bounded = Math.max(min, Math.min(max, numeric));
    update(key, bounded);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Settings</h3>
        
        {/* API Key Section */}
        <div style={{ 
          marginBottom: 24, 
          padding: 16, 
          background: providerKeyConnected ? "rgba(152, 206, 0, 0.1)" : "rgba(201, 107, 107, 0.1)",
          borderRadius: 12,
          border: `1px solid ${providerKeyConnected ? "rgba(152, 206, 0, 0.3)" : "rgba(201, 107, 107, 0.3)"}`
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                API Key ({providerLabel})
              </div>
              <div style={{ fontSize: 12, color: "var(--muted)" }}>
                {providerKeyConnected ? "API key is configured" : "API key not added"}
              </div>
            </div>
            <button
              className="btn-primary"
              onClick={() => {
                onClose();
                onAddApiKey?.(provider);
              }}
              style={{ padding: "8px 16px", fontSize: 13 }}
            >
              {providerKeyConnected ? "Update API Key" : "Add API Key"}
            </button>
          </div>
        </div>

        <div className="settings-grid">
          <label>
            Model Provider
            <select value={provider} onChange={(e) => update("modelProvider", e.target.value)}>
              <option value="anthropic">Anthropic Claude</option>
              <option value="openai">OpenAI ChatGPT</option>
              <option value="gemini">Google Gemini</option>
              <option value="zai">Z.ai GLM</option>
            </select>
          </label>
          <label>
            Model
            <input
              value={form.model || ""}
              onChange={(e) => update("model", e.target.value)}
              placeholder={
                provider === "anthropic"
                  ? "claude-sonnet-4-5-20250929"
                  : provider === "openai"
                    ? "gpt-5.2"
                    : provider === "gemini"
                      ? "gemini-3-flash-preview"
                      : "glm-4.7"
              }
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
          <label>
            Verify Every N Steps
            <input
              type="number"
              min={1}
              max={12}
              step={1}
              value={form.verificationEveryNSteps ?? form.screenshotFrequency ?? 2}
              onChange={(e) => updateNumber("verificationEveryNSteps", e.target.value, 1, 12)}
            />
          </label>
          <label>
            LLM Timeout (seconds)
            <input
              type="number"
              min={5}
              max={300}
              step={5}
              value={form.llmTimeoutSeconds ?? 60}
              onChange={(e) => updateNumber("llmTimeoutSeconds", e.target.value, 5, 300)}
            />
          </label>
          <label>
            Max No-Advance Checks
            <input
              type="number"
              min={1}
              max={12}
              step={1}
              value={form.maxStagnantSteps ?? 4}
              onChange={(e) => updateNumber("maxStagnantSteps", e.target.value, 1, 12)}
            />
          </label>
          <label>
            Max Stuck Signals
            <input
              type="number"
              min={1}
              max={8}
              step={1}
              value={form.maxStuckSignals ?? 2}
              onChange={(e) => updateNumber("maxStuckSignals", e.target.value, 1, 8)}
            />
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
