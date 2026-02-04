import React, { useState, useEffect } from "react";

export default function ApiKeyModal({ onSaved, onSkip }) {
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [skipped, setSkipped] = useState(false);
  const [bridgeAvailable, setBridgeAvailable] = useState(!!window.navai?.apiKeySet);

  // Check if bridge is available when component mounts
  useEffect(() => {
    const checkBridge = () => {
      const available = !!window.navai?.apiKeySet;
      setBridgeAvailable(available);
      return available;
    };

    // Check immediately
    if (checkBridge()) return;

    // If not available, wait a bit and retry (bridge might still be loading)
    let attempts = 0;
    const maxAttempts = 10;
    
    const retryTimer = setInterval(() => {
      attempts++;
      if (checkBridge() || attempts >= maxAttempts) {
        clearInterval(retryTimer);
        if (!window.navai?.apiKeySet && attempts >= maxAttempts) {
          console.error("Preload bridge not available after retries. window.navai:", window.navai);
        }
      }
    }, 100);

    return () => clearInterval(retryTimer);
  }, []);

  const save = async () => {
    setError("");
    
    if (!key.trim()) {
      setError("Please enter a key");
      return;
    }
    
    // Wait for bridge if not available yet - try multiple times
    let bridgeReady = !!window.navai?.apiKeySet;
    if (!bridgeReady) {
      // Try waiting up to 2 seconds for bridge to be ready
      for (let i = 0; i < 20; i++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        if (window.navai?.apiKeySet) {
          bridgeReady = true;
          break;
        }
      }
    }
    
    if (!bridgeReady) {
      console.error("Bridge check failed. window.navai:", window.navai);
      setError("Preload bridge not available. Please refresh the page (Cmd+R) or restart the app.");
      return;
    }
    
    try {
      console.log("Attempting to save API key...");
      const result = await window.navai.apiKeySet(key.trim());
      console.log("API key save result:", result);
      
      if (result?.ok) {
        console.log("API key saved successfully, restarting Python...");
        await window.navai?.pythonRestart();
        console.log("Python restarted, calling onSaved callback");
        onSaved?.();
      } else {
        const errorMsg = result?.error || "Could not save key";
        console.error("Failed to save API key:", errorMsg);
        setError(errorMsg);
      }
    } catch (err) {
      console.error("Error saving API key:", err);
      setError(`Error saving key: ${err.message || String(err)}`);
    }
  };

  const handleSkip = () => {
    setSkipped(true);
    onSkip?.();
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
          onKeyDown={(e) => {
            if (e.key === "Enter" && key.trim()) {
              save();
            }
          }}
          autoFocus
          style={{ width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid rgba(132,145,163,0.35)" }}
        />
        {!bridgeAvailable && (
          <div style={{ 
            color: "#c96b6b", 
            fontSize: 11, 
            marginTop: 8,
            padding: 8,
            background: "rgba(201, 107, 107, 0.1)",
            borderRadius: 6
          }}>
            Waiting for bridge to initialize...
          </div>
        )}
        {error && <div style={{ color: "#c96b6b", fontSize: 12, marginTop: 6 }}>{error}</div>}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
          <button className="btn-outline" onClick={handleSkip}>Skip for now</button>
          <button 
            className="btn-primary" 
            onClick={save} 
            disabled={!key.trim()}
            style={{ opacity: !key.trim() ? 0.5 : 1, cursor: !key.trim() ? 'not-allowed' : 'pointer' }}
          >
            Save Key
          </button>
        </div>
      </div>
    </div>
  );
}
