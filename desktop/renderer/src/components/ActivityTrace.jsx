import React, { useState } from "react";

export default function ActivityTrace({ steps, onOpen }) {
  const [open, setOpen] = useState(true);
  if (!steps?.length) return null;

  return (
    <div className="activity">
      <div className="activity-toggle" onClick={() => setOpen((s) => !s)}>
        {open ? "Hide" : "Show"} live activity
      </div>
      {open
        ? steps.map((step, idx) => (
            <div className="step-card" key={step?.id ?? `step-${idx}`}>
              <div>
                <div className="step-title">{step.title}</div>
                {step.caption && <div className="step-caption">{step.caption}</div>}
                {step.screenshots?.length ? (
                  <div className="thumb-grid">
                    {step.screenshots.map((shot, idx) => (
                      <img
                        key={`${step.id}-${idx}`}
                        className="thumb"
                        src={
                          typeof window.navai?.fileUrl === "function"
                            ? window.navai.fileUrl(shot.path)
                            : shot.path
                              ? `file://${shot.path}`
                              : ""
                        }
                        alt="screenshot"
                        onClick={() => onOpen(shot)}
                      />
                    ))}
                  </div>
                ) : null}
              </div>
              <div className="step-time">{step.timestamp}</div>
            </div>
          ))
        : null}
    </div>
  );
}
