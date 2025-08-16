"use client";

import React from "react";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelect }) => {
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSelect(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      onImageSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={() => document.getElementById("file-input")?.click()}
      style={{
        border: "2px dashed #ccc",
        borderRadius: "10px",
        padding: "1rem",
        textAlign: "center",
        cursor: "pointer",
        transition: "border-color 0.2s ease-in-out",
        marginBottom: "1.5rem",
      }}
    >
      <div style={{ fontSize: "2rem", marginBottom: ".5rem" }}>📷</div>
      <p style={{ margin: "0.25rem 0", fontWeight: 500 }}>
        Click or drag to upload
      </p>
      <p style={{ margin: ".25rem 0", fontSize: "0.9rem", color: "#666" }}>
        Image file (e.g., cat or dog)
      </p>
      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        style={{ display: "none" }}
      />
    </div>
  );
};
