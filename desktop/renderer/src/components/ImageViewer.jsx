import React, { useState } from "react";
import { resolveScreenshotPath } from "./utils.js";

export default function ImageViewer({ image, onClose, screenshotsDir }) {
  const [zoom, setZoom] = useState(1);

  if (!image) return null;

  const fullPath = resolveScreenshotPath(image.path, screenshotsDir);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="viewer">
          <img
            src={fullPath ? `file://${fullPath}` : ""}
            alt="screenshot"
            style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}
          />
          <div className="viewer-actions">
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-outline" onClick={() => setZoom((z) => Math.max(0.6, z - 0.2))}>-</button>
              <button className="btn-outline" onClick={() => setZoom((z) => Math.min(2.4, z + 0.2))}>+</button>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-outline" onClick={() => window.navai?.openFile(fullPath || image.path)}>Open File</button>
              <button className="btn-outline" onClick={onClose}>Close</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
