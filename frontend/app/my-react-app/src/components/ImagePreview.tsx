"use client";
import React, { useRef } from "react";
import { Card } from "./Card";

interface ImagePreviewProps {
  imagePreview: string;
  onClear: () => void;
  onReplace: (file: File) => void;
}

export const ImagePreview: React.FC<ImagePreviewProps> = ({
  imagePreview,
  onReplace,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onReplace(file);
    }
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <Card>
        <div
          style={{
            position: "relative",
            cursor: "pointer",
            textAlign: "center",
          }}
          onClick={() => inputRef.current?.click()}
        >
          <img
            src={imagePreview || "/placeholder.svg"}
            alt="Selected image"
            style={{
              width: "100%",
              maxHeight: "300px",
              objectFit: "cover",
              borderRadius: "10px",
              display: "block",
            }}
          />
          <input
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            ref={inputRef}
            style={{ display: "none" }}
          />
        </div>
      </Card>
    </div>
  );
};
