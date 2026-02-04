import React, { useState } from "react";

export default function ImageViewer({ image, onClose }) {
  const [zoom, setZoom] = useState(1);

  if (!image) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="viewer">
          <img
            src={
              typeof window.navai?.fileUrl === "function"
                ? window.navai.fileUrl(image.path)
                : image.path
                  ? `file://${image.path}`
                  : ""
            }
            alt="screenshot"
            style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}
          />
          <div className="viewer-actions">
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-outline" onClick={() => setZoom((z) => Math.max(0.6, z - 0.2))}>-</button>
              <button className="btn-outline" onClick={() => setZoom((z) => Math.min(2.4, z + 0.2))}>+</button>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-outline" onClick={() => window.navai?.openFile(image.path)}>Open File</button>
              <button className="btn-outline" onClick={onClose}>Close</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
