import React, { useEffect, useMemo, useRef, useState } from "react";
import ActivityTrace from "./components/ActivityTrace.jsx";
import SettingsModal from "./components/SettingsModal.jsx";
import ApiKeyModal from "./components/ApiKeyModal.jsx";
import ImageViewer from "./components/ImageViewer.jsx";
import EntranceScreen from "./components/EntranceScreen.jsx";
import Toast from "./components/Toast.jsx";
import { nowLabel, uid } from "./components/utils.js";
import logoUrl from "./assets/logo.png";

class AppErrorBoundary extends React.Component {
  state = { hasError: false, error: null };
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("App error:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="app" style={{ padding: 24, display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", flexDirection: "column", gap: 12 }}>
          <p style={{ color: "#666" }}>Something went wrong while showing the response.</p>
          <button
            className="btn-primary"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const DEFAULT_CONVO = {
  id: uid(),
  title: "New conversation",
  updatedAt: nowLabel(),
  messages: [],
  agentMode: "unknown"
};

const sanitizeTitle = (title) => {
  if (title == null) return "New conversation";
  let s = String(title).trim();
  // Strip surrounding quotes.
  s = s.replace(/^["'“”]+|["'“”]+$/g, "").trim();
  // Strip markdown heading/bullet prefixes (common from LLMs).
  s = s.replace(/^\s*#{1,6}\s*/, "");
  s = s.replace(/^\s*[-*]\s+/, "");
  // Normalize whitespace.
  s = s.replace(/\s+/g, " ").trim();
  return s || "New conversation";
};

const PROVIDER_LABELS = {
  anthropic: "Anthropic Claude",
  openai: "OpenAI ChatGPT",
  gemini: "Google Gemini",
  zai: "Z.ai GLM"
};

const QUICK_MODELS = [
  { provider: "anthropic", model: "claude-sonnet-4-5-20250929", label: "Claude Sonnet 4.5" },
  { provider: "openai", model: "gpt-5.2", label: "ChatGPT GPT-5.2" },
  { provider: "gemini", model: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview" },
  { provider: "zai", model: "glm-4.7", label: "Z.ai GLM-4.7" }
];

export default function App() {
  const [conversations, setConversations] = useState(() => {
    try {
      const stored = localStorage.getItem("navai_conversations");
      const parsed = stored ? JSON.parse(stored) : [DEFAULT_CONVO];
      const list = Array.isArray(parsed) && parsed.length ? parsed : [DEFAULT_CONVO];
      // Clean up any previously-saved titles (e.g. starting with '#').
      return list.map((c) => ({
        ...c,
        title: sanitizeTitle(c?.title),
        agentMode: c?.agentMode || "unknown"
      }));
    } catch {
      return [DEFAULT_CONVO];
    }
  });
  const [activeId, setActiveId] = useState(conversations[0]?.id);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [composer, setComposer] = useState("");
  const [running, setRunning] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [viewer, setViewer] = useState(null);
  const [apiKeyStatus, setApiKeyStatus] = useState({ connected: false });
  const [settings, setSettings] = useState({
    modelProvider: "anthropic",
    model: "claude-sonnet-4-5-20250929",
    screenshotFrequency: 2,
    verificationEveryNSteps: 2,
    saveScreenshots: true,
    allowSendScreenshots: true,
    dryRun: false,
    llmTimeoutSeconds: 60,
    maxStagnantSteps: 4,
    maxStuckSignals: 2
  });
  const [search, setSearch] = useState("");
  const [ws, setWs] = useState(null);
  const [wsStatus, setWsStatus] = useState("connecting");
  const [wsError, setWsError] = useState("");
  const [showEntrance, setShowEntrance] = useState(true);
  const [toast, setToast] = useState(null);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKeyModalProvider, setApiKeyModalProvider] = useState("");
  const [apiKeyModalSkipped, setApiKeyModalSkipped] = useState(false);
  const [storagePaths, setStoragePaths] = useState({ conversationsDir: "", screenshotsDir: "" });
  const [showModelMenu, setShowModelMenu] = useState(false);

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId),
    [conversations, activeId]
  );

  const threadEndRef = useRef(null);
  const activeProvider = settings.modelProvider || "anthropic";
  const activeProviderConnected = Boolean(apiKeyStatus?.providers?.[activeProvider]);

  useEffect(() => {
    localStorage.setItem("navai_conversations", JSON.stringify(conversations));
  }, [conversations]);

  const lastMessageKey = useMemo(() => {
    const msgs = activeConversation?.messages;
    if (!Array.isArray(msgs) || !msgs.length) return "";
    const last = msgs[msgs.length - 1];
    const id = last?.id ?? "last";
    const contentLen = typeof last?.content === "string" ? last.content.length : String(last?.content ?? "").length;
    return `${activeConversation?.id ?? ""}:${id}:${contentLen}`;
  }, [activeConversation?.id, activeConversation?.messages]);

  // Keep scrolling as new words stream in from message deltas.
  useEffect(() => {
    // Always scroll to bottom when messages change, not just when running
    requestAnimationFrame(() => {
      threadEndRef.current?.scrollIntoView({ block: "end", behavior: "auto" });
    });
  }, [lastMessageKey]);

  useEffect(() => {
    const refreshApiKeyStatus = async () => {
      try {
        const status = await window.navai?.apiKeyStatus();
        setApiKeyStatus(status || { connected: false, providers: {} });
      } catch (err) {
        console.error("Error checking API key status:", err);
        setApiKeyStatus({ connected: false, providers: {} });
      }
    };

    const checkApiKey = async () => {
      try {
        const status = await window.navai?.apiKeyStatus();
        console.log("Initial API key status:", status);
        setApiKeyStatus(status || { connected: false, providers: {} });
      } catch (err) {
        console.error("Error checking API key status:", err);
        setApiKeyStatus({ connected: false, providers: {} });
      }
    };

    checkApiKey();
    window.navai?.settingsGet().then((s) => s && setSettings((prev) => ({ ...prev, ...s })));
    window.navai?.storagePaths?.().then((p) => p && setStoragePaths((prev) => ({ ...prev, ...p })));
    window.navai?.apiKeyStatus && refreshApiKeyStatus();
  }, []);

  useEffect(() => {
    let socket;
    let retryTimer;
    let initialDelayTimer;
    let canceled = false;

    const connect = () => {
      if (canceled) return;
      setWsStatus("connecting");
      socket = new WebSocket("ws://127.0.0.1:8765");
      socket.onopen = () => {
        setWs(socket);
        setWsStatus("connected");
        setWsError("");
      };
      socket.onclose = () => {
        setWs(null);
        if (!canceled) {
          setWsStatus("disconnected");
          retryTimer = setTimeout(connect, 1500);
        }
      };
      socket.onerror = () => {
        setWs(null);
        if (!canceled) {
          setWsStatus("error");
          setWsError((prev) => prev || "Agent server not reachable. If you just started the app, wait a moment—Python may still be starting. If it keeps failing, check the terminal where you ran the app for Python errors (e.g. missing websockets, wrong path to agent_server.py).");
        }
      };
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          handleAgentEvent(payload);
        } catch (err) {
          console.error("Agent message parse error:", err);
          // Don't crash the app on bad WebSocket data
        }
      };
    };

    // Give the Python server time to start (Electron spawns it when the app loads).
    initialDelayTimer = setTimeout(connect, 1200);

    return () => {
      canceled = true;
      clearTimeout(retryTimer);
      clearTimeout(initialDelayTimer);
      socket?.close();
    };
  }, []);

  const handleAgentEvent = (payload) => {
    if (!payload || typeof payload.type !== "string") return;
    if (payload.type === "status") {
      setRunning(payload.state === "running");
      window.navai?.setAgentWindowMode?.(payload.state === "running");
      return;
    }

    if (!activeConversation) return;

    const targetConvoId = payload.conversationId ?? activeId;

    setConversations((prev) => {
      return prev.map((convo) => {
        if (payload.type === "title") {
          if (convo.id !== payload.conversationId) return convo;
          return { ...convo, title: sanitizeTitle(payload.title), updatedAt: nowLabel() };
        }
        if (payload.type === "decision_mode") {
          return {
            ...convo,
            agentMode: payload.mode || "unknown",
            updatedAt: nowLabel()
          };
        }
        if (convo.id !== targetConvoId) return convo;
        const messages = [...convo.messages];

        if (payload.type === "message_delta") {
          const role = payload.role === "assistant" || payload.role === "user" ? payload.role : "assistant";
          const messageId = payload.messageId ?? uid();
          const delta = typeof payload.delta === "string" ? payload.delta : String(payload.delta ?? "");
          const last = messages[messages.length - 1];
          if (!last || last.role !== role || last.id !== messageId) {
            messages.push({
              id: messageId,
              role,
              content: delta,
              timestamp: nowLabel(),
              activity: { steps: [] }
            });
          } else {
            messages[messages.length - 1] = {
              ...last,
              content: (last.content ?? "") + delta
            };
          }
        }

        if (payload.type === "step") {
          const last = messages[messages.length - 1];
          if (last) {
            const stepId = payload.stepId ?? uid();
            const steps = [...(last.activity?.steps ?? []), {
              id: stepId,
              title: payload.title ?? "",
              caption: payload.caption ?? "",
              timestamp: payload.timestamp ?? nowLabel(),
              screenshots: []
            }];
            messages[messages.length - 1] = {
              ...last,
              activity: { ...last.activity, steps }
            };
          }
        }

        if (payload.type === "screenshot") {
          const last = messages[messages.length - 1];
          if (last?.activity?.steps?.length) {
            const stepIndex = last.activity.steps.findIndex((s) => s.id === payload.stepId);
            const steps = [...last.activity.steps];
            const step = stepIndex >= 0 ? steps[stepIndex] : steps[steps.length - 1];
            if (step) {
              const idx = stepIndex >= 0 ? stepIndex : steps.length - 1;
              steps[idx] = {
                ...step,
                screenshots: [...(step.screenshots ?? []), {
                  path: payload.path,
                  caption: payload.caption,
                  timestamp: payload.timestamp
                }]
              };
              messages[messages.length - 1] = {
                ...last,
                activity: { ...last.activity, steps }
              };
            }
          }
        }

        if (payload.type === "tool") {
          const last = messages[messages.length - 1];
          if (last?.activity?.steps?.length) {
            const steps = [...last.activity.steps];
            const step = steps[steps.length - 1];
            if (step) {
              steps[steps.length - 1] = { ...step, caption: payload.caption ?? step.caption };
              messages[messages.length - 1] = {
                ...last,
                activity: { ...last.activity, steps }
              };
            }
          }
        }

        return { ...convo, messages, updatedAt: nowLabel() };
      });
    });
  };

  const titleFromPrompt = (text) => {
    const words = text.trim().split(/\s+/).slice(0, 4);
    if (!words.length) return "New conversation";
    const base = sanitizeTitle(words.join(" "));
    const t = base.length > 40 ? `${base.slice(0, 37)}...` : base;
    return t.endsWith("...") ? t : `${t}...`;
  };

  const sendPrompt = async () => {
    if (!composer.trim() || !activeConversation) return;
    if (!ws) {
      setWsStatus("error");
      setWsError((prev) => prev || "Agent server not running. Please check Python setup.");
      return;
    }
    if (!activeProviderConnected) {
      setApiKeyModalProvider(activeProvider);
      setShowApiKeyModal(true);
      setApiKeyModalSkipped(false);
      setToast({ message: `Add an API key for ${PROVIDER_LABELS[activeProvider] || activeProvider}`, type: "error" });
      return;
    }
    if (activeConversation.title === "New conversation") {
      const nextTitle = titleFromPrompt(composer);
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeConversation.id ? { ...c, title: nextTitle, updatedAt: nowLabel() } : c
        )
      );
    }
    const messageId = uid();
    const newUserMsg = {
      id: messageId,
      role: "user",
      content: composer.trim(),
      timestamp: nowLabel()
    };

    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeConversation.id
          ? { ...c, messages: [...c.messages, newUserMsg], updatedAt: nowLabel() }
          : c
      )
    );

    ws.send(
      JSON.stringify({
        type: "run",
        conversationId: activeConversation.id,
        prompt: composer.trim(),
        settings,
        requestTitle: activeConversation.title === "New conversation"
      })
    );

    setComposer("");
  };

  const stopAgent = () => {
    if (!ws || !activeConversation) return;
    ws.send(
      JSON.stringify({
        type: "stop",
        conversationId: activeConversation.id
      })
    );
  };

  const createConversation = () => {
    const next = {
      id: uid(),
      title: "New conversation",
      updatedAt: nowLabel(),
      messages: [],
      agentMode: "unknown"
    };
    setConversations((prev) => [next, ...prev]);
    setActiveId(next.id);
  };

  const deleteConversation = (id) => {
    setConversations((prev) => {
      const remaining = prev.filter((c) => c.id !== id);
      if (!remaining.length) {
        const next = { ...DEFAULT_CONVO, id: uid(), updatedAt: nowLabel() };
        setActiveId(next.id);
        return [next];
      }
      if (id === activeId) {
        setActiveId(remaining[0].id);
      }
      return remaining;
    });
  };

  const filteredConversations = conversations.filter((c) =>
    c.title.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <AppErrorBoundary>
      <div className="app">
        <div className="app-content">
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: 38, /* Traffic light safe area */
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            WebkitAppRegion: 'drag',
            zIndex: 9999,
            pointerEvents: 'none',
            fontSize: 12,
            fontWeight: 600,
            color: 'rgba(0,0,0,0.4)'
          }}>
            <button
              className="window-toggle-btn no-drag"
              onClick={() => setSidebarCollapsed((s) => !s)}
              title="Toggle Sidebar"
              style={{
                pointerEvents: 'auto',
                position: 'absolute',
                left: 85, /* Avoid traffic lights */
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'transparent',
                border: 'none',
                cursor: 'default',
                padding: 4,
                borderRadius: 4,
                color: 'rgba(0,0,0,0.5)'
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 6H20M4 12H20M4 18H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            Nav
          </div>
          <div className={`left-column ${sidebarCollapsed ? "collapsed" : ""}`} style={{ width: sidebarCollapsed ? 68 : sidebarWidth }}>
            <button className="new-chat-btn" onClick={createConversation} title="New Chat">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 4V20M4 12H20" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span>New Chat</span>
            </button>

            <aside
              className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}
              style={{ width: "100%" }}
            >
              <div className="sidebar-header drag-region">
                {/* Logo removed from here */}
                {/* Logo removed from here */}
              </div>

              {!sidebarCollapsed && (
                <>
                  <div className="convo-list">
                    {filteredConversations.map((c) => (
                      <div
                        key={c.id}
                        className={`convo-item ${c.id === activeId ? "active" : ""}`}
                        onClick={() => setActiveId(c.id)}
                      >
                        <div className="convo-title-row">
                          <div className="convo-title">{c.title}</div>
                        </div>
                        <button
                          className="convo-delete no-drag"
                          title="Delete chat"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteConversation(c.id);
                          }}
                        >
                          <svg viewBox="0 0 24 24" aria-hidden="true" fill="#dc2626">
                            <path d="M9 3h6l1 2h4v2H4V5h4l1-2z" />
                            <path d="M6 9h12l-1 11H7L6 9z" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                  <div className="sidebar-footer">
                    {storagePaths?.conversationsDir || storagePaths?.screenshotsDir ? (
                      <div className="storage-block">
                        <div className="storage-title">Storage</div>
                        <button
                          className="storage-link"
                          type="button"
                          onClick={() => window.navai?.openStorageDir?.("conversations")}
                          title="Open conversations folder"
                        >
                          <span className="storage-label">Conversations (.txt)</span>
                          <span className="storage-path">{storagePaths.conversationsDir || "—"}</span>
                        </button>
                        <button
                          className="storage-link"
                          type="button"
                          onClick={() => window.navai?.openStorageDir?.("screenshots")}
                          title="Open screenshots folder"
                        >
                          <span className="storage-label">Screenshots</span>
                          <span className="storage-path">{storagePaths.screenshotsDir || "—"}</span>
                        </button>
                      </div>
                    ) : null}
                    <button
                      className={`btn-outline ${sidebarCollapsed ? "icon-only" : ""}`}
                      onClick={() => setSettingsOpen(true)}
                      title="Settings"
                      style={sidebarCollapsed ? {
                        padding: 10,
                        width: 36,
                        height: 36,
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        borderRadius: '50%',
                        margin: '0 auto'
                      } : {}}
                    >
                      {sidebarCollapsed ? (
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M19.4 15C19.743 14.5165 20.0345 13.9922 20.264 13.447C20.6698 12.4828 20.6698 11.5172 20.264 10.553C20.0345 10.0078 19.743 9.48353 19.4 9M19.4 15C19.4 15 18 14.4 17.2 14.4C16.5 14.4 15.8 14.8 15.6 15.6C15.4 16.4 15.8 17 15.8 17C15.8 17 16.8 18 16.8 18C16.8 18 15.8 19.2 14.8 19.8C14.8 19.8 14 19 13.2 19C12.4 19 11.6 19 10.8 19C10 19 9.2 19.8 9.2 19.8C9.2 19.8 8.2 18.6 8.2 18.6C8.2 18.6 9.2 17.6 9.2 17.6C9.2 16.8 9.6 16.2 9.4 15.4C9.2 14.6 8.5 14.2 7.8 14.2C7 14.2 5.6 15 5.6 15M19.4 9C19.4 9 18 9.6 17.2 9.6C16.5 9.6 15.8 9.2 15.6 8.4C15.4 7.6 15.8 7 15.8 7C15.8 7 16.8 6 16.8 6C16.8 6 15.8 4.8 14.8 4.2C14.8 4.2 14 5 13.2 5C12.4 5 11.6 5 10.8 5C10 5 9.2 4.2 9.2 4.2C9.2 4.2 8.2 5.4 8.2 5.4C8.2 5.4 9.2 6.4 9.2 6.4C9.2 7.2 9.6 7.8 9.4 8.6C9.2 9.4 8.5 9.8 7.8 9.8C7 9.8 5.6 9 5.6 9M5.6 15C5.25701 14.5165 4.96547 13.9922 4.736 13.447C4.33022 12.4828 4.33022 11.5172 4.736 10.553C4.96547 10.0078 5.25701 9.48353 5.6 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      ) : "Settings"}
                    </button>
                  </div>
                </>
              )}
              {!sidebarCollapsed && (
                <div
                  className="sidebar-resizer"
                  onMouseDown={(e) => {
                    const startX = e.clientX;
                    const startWidth = sidebarWidth;
                    const onMove = (moveEvent) => {
                      const next = Math.min(420, Math.max(240, startWidth + moveEvent.clientX - startX));
                      setSidebarWidth(next);
                    };
                    const onUp = () => {
                      window.removeEventListener("mousemove", onMove);
                      window.removeEventListener("mouseup", onUp);
                    };
                    window.addEventListener("mousemove", onMove);
                    window.addEventListener("mouseup", onUp);
                  }}
                />
              )}
            </aside>
          </div>

          <section className="main">
            <div className="header border border-b rounded-full drag-region">
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div>
                  <div className="header-title">{activeConversation?.title || "Conversation"}</div>
                  <div className="header-status-wrap">
                    <div className="header-status">{running ? "Running" : "Idle"}</div>
                    <div className={`mode-badge mode-${activeConversation?.agentMode || "unknown"}`}>
                      {activeConversation?.agentMode === "fallback_with_element_ids"
                        ? "Fallback IDs"
                        : activeConversation?.agentMode === "raw_screenshot_only"
                          ? "Raw Screenshot"
                          : "Mode Unknown"}
                    </div>
                  </div>
                </div>
              </div>
              {running ? (
                <button className="btn-outline no-drag" onClick={stopAgent}>Stop</button>
              ) : null}
            </div>
            {wsStatus !== "connected" && (
              <div className="banner">
                Agent server: {wsStatus}. {wsError || "Attempting to reconnect."}
              </div>
            )}
            {!activeProviderConnected && (
              <div className="banner">
                API key not added for {PROVIDER_LABELS[activeProvider] || activeProvider}.
              </div>
            )}

            <div className="thread">
              {activeConversation?.messages.map((m, idx) => (
                <div key={m?.id ?? `msg-${idx}`} className={`message ${m?.role ?? "assistant"}`}>
                  <div className={`bubble ${m?.role ?? "assistant"}`}>
                    <div>{typeof m?.content === "string" ? m.content : String(m?.content ?? "")}</div>
                    {m.role === "assistant" && m.activity?.steps?.length ? (
                      <ActivityTrace
                        steps={m.activity.steps}
                        onOpen={(shot) => setViewer(shot)}
                        screenshotsDir={storagePaths.screenshotsDir}
                      />
                    ) : null}
                  </div>
                </div>
              ))}
              <div ref={threadEndRef} style={{ height: 1 }} />
            </div>

            <div className="composer">
              <div className="composer-input">
                <div className="model-selector-wrapper">
                  <button
                    className="model-selector-btn"
                    onClick={() => setShowModelMenu(!showModelMenu)}
                    title={settings.model}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M6 9L12 15L18 9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                  {showModelMenu && (
                    <div className="model-menu">
                      {QUICK_MODELS.map((preset) => (
                        <button
                          key={`${preset.provider}:${preset.model}`}
                          className={`model-option ${settings.modelProvider === preset.provider && settings.model === preset.model ? "active" : ""}`}
                          onClick={() => {
                            const next = { ...settings, modelProvider: preset.provider, model: preset.model };
                            setSettings(next);
                            setShowModelMenu(false);
                            window.navai?.settingsSet && window.navai.settingsSet(next);
                            if (!apiKeyStatus?.providers?.[preset.provider]) {
                              setApiKeyModalProvider(preset.provider);
                              setShowApiKeyModal(true);
                              setApiKeyModalSkipped(false);
                            }
                          }}
                        >
                          {preset.label}
                          {settings.modelProvider === preset.provider && settings.model === preset.model && (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="check-icon">
                              <path d="M20 6L9 17L4 12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <input
                  type="text"
                  placeholder="Tell the agent what to do..."
                  value={composer}
                  onChange={(e) => setComposer(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      if (running) {
                        stopAgent();
                      } else {
                        void sendPrompt();
                      }
                    }
                  }}
                />
                <button
                  className={`btn-primary send-inline ${running ? "stop-btn" : ""}`}
                  onClick={running ? stopAgent : () => void sendPrompt()}
                  disabled={!ws && !running}
                  title={running ? "Stop" : "Send"}
                >
                  {running ? (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <rect x="6" y="6" width="12" height="12" rx="1" fill="currentColor" />
                    </svg>
                  ) : (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M7 17L17 7M17 7H7M17 7V17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </section>

          {settingsOpen && (
            <SettingsModal
              settings={settings}
              apiKeyStatus={apiKeyStatus}
              onClose={() => setSettingsOpen(false)}
              onSave={async (next) => {
                try {
                  setSettings(next);
                  const result = await window.navai?.settingsSet(next);
                  if (result?.ok !== false) {
                    setToast({ message: "Settings saved", type: "success" });
                    setSettingsOpen(false);
                  } else {
                    setToast({ message: "Unable to save settings", type: "error" });
                  }
                } catch (error) {
                  setToast({ message: "Unable to save settings", type: "error" });
                }
              }}
              onAddApiKey={(provider) => {
                setApiKeyModalProvider(provider || settings.modelProvider || activeProvider || "anthropic");
                setShowApiKeyModal(true);
                setApiKeyModalSkipped(false);
              }}
            />
          )}

          {(showApiKeyModal || (!activeProviderConnected && !apiKeyModalSkipped)) && (
            <ApiKeyModal
              provider={apiKeyModalProvider || activeProvider}
              onSaved={async () => {
                const status = await window.navai?.apiKeyStatus();
                setApiKeyStatus(status || { connected: false, providers: {} });
                setShowApiKeyModal(false);
                setApiKeyModalSkipped(true);
                setToast({ message: "API key saved", type: "success" });
              }}
              onSkip={() => {
                setApiKeyModalSkipped(true);
                setShowApiKeyModal(false);
              }}
            />
          )}

          {toast && (
            <div className="toast-container">
              <Toast
                message={toast.message}
                type={toast.type}
                onClose={() => setToast(null)}
              />
            </div>
          )}

          {viewer && (
            <ImageViewer
              image={viewer}
              onClose={() => setViewer(null)}
              screenshotsDir={storagePaths.screenshotsDir}
            />
          )}

          {showEntrance && (
            <EntranceScreen onComplete={() => setShowEntrance(false)} />
          )}
        </div>
      </div>
    </AppErrorBoundary>
  );
}
