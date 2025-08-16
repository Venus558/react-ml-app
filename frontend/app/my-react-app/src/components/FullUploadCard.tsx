"use client";
import React from "react";
import { Card } from "./Card";
import { ImageUpload } from "./ImageUpload";
import { ImagePreview } from "./ImagePreview";
import { SubmitButton } from "./SubmitButton";

interface FullUploadCardProps {
  selectedImage: File | null;
  imagePreview: string;
  onImageSelect: (file: File) => void;
  onClear: () => void;
  onSubmit: () => void;
  loading: boolean;
}

export const FullUploadCard: React.FC<FullUploadCardProps> = ({
  selectedImage,
  imagePreview,
  onImageSelect,
  onClear,
  onSubmit,
  loading,
}) => {
  return (
    <div className="">
      <h1
        style={{
          fontSize: "3rem", // same as Tailwind's text-4xl
          fontWeight: "bold",
          color: "#1a1a1a",
          marginBottom: "1rem",
        }}
      >
        Upload Image
      </h1>
      <Card>
        {!selectedImage ? (
          <ImageUpload onImageSelect={onImageSelect} />
        ) : (
          <ImagePreview
            imagePreview={imagePreview}
            onClear={onClear}
            onReplace={onImageSelect}
          />
        )}
        <div className="flex justify-center mt-4">
          <SubmitButton onClick={onSubmit} disabled={loading} loading={loading}>
            Submit
          </SubmitButton>
        </div>
      </Card>
    </div>
  );
};
