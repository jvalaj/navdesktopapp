import React, { useState } from "react";
import { resolveScreenshotPath } from "./utils.js";

export default function ActivityTrace({ steps, onOpen, screenshotsDir }) {
  const [open, setOpen] = useState(true);
  if (!steps?.length) return null;

  return (
    <div className="activity">
      <div className="activity-toggle" onClick={() => setOpen((s) => !s)}>
        {open ? "Hide" : "Show"} live activity
      </div>
      {open
        ? steps.map((step, stepIdx) => (
            <div className="step-card" key={`step-${stepIdx}`}>
              <div>
                <div className="step-title">{step.title}</div>
                {step.caption && <div className="step-caption">{step.caption}</div>}
                {step.screenshots?.length ? (
                  <div className="thumb-grid">
                    {step.screenshots.map((shot, shotIdx) => {
                      const fullPath = resolveScreenshotPath(shot.path, screenshotsDir);
                      return (
                        <img
                          key={`step-${stepIdx}-shot-${shotIdx}`}
                          className="thumb"
                          src={fullPath ? `file://${fullPath}` : ""}
                          alt="screenshot"
                          onClick={() => onOpen(shot)}
                        />
                      );
                    })}
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
