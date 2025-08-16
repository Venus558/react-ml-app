"use client";
import "./App.css";
import { useEffect, useState } from "react";
import { BackgroundContainer } from "./components/BackgroundContainer";
import axios from "axios";
import { FullUploadCard } from "./components/FullUploadCard";
import { PredictionBreakdwon } from "./components/PredictionBreakdown";
import { FinalPrediction } from "./components/FinalPrediction";
import SectionBanner from "./components/SectionBanner";

export default function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
  };

  const handleClear = () => {
    setSelectedImage(null);
    setImagePreview("");
  };

  const [predictions, setPredictions] = useState<
    { label: string; confidence: number }[]
  >([]);

  const handleSubmit = async () => {
    if (!selectedImage) {
      alert("Please upload an image first.");
      setHasSubmitted(true);
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      setLoading(true);
      const res = await axios.post("http://localhost:5000/classify", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPredictions(res.data.predictions);
      setHasSubmitted(true);
    } catch (err) {
      console.error("Upload error:", err);
      alert("Submission Error");
    } finally {
      setLoading(false);
    }
  };

  const [hasSubmitted, setHasSubmitted] = useState(false);

  const [isMobile, setIsMobile] = useState(false);

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return (
    <BackgroundContainer>
      <div
        style={{
          width: "100%",
          maxWidth: "1200px",
          display: "block",
          flexDirection: "column",
          alignItems: "center",
          gap: "2rem",
          padding: "1rem",
          margin: "0 auto",
        }}
      >
        <SectionBanner
          title="Neural Network Classifier"
          subtitle="This AI model uses deep learning to detect whether an image contains a dog, a cat, or neither."
          variant="accent"
        />

        {hasSubmitted ? (
          <div
            style={{
              display: "flex",
              flexDirection: isMobile ? "column" : "row",
              width: "100%",
              maxWidth: "1200px",
              justifyContent: "center",
              alignItems: isMobile ? "center" : "stretch",
              gap: "1.5rem",
              padding: "1rem",
              boxSizing: "border-box",
            }}
          >
            <div
              style={{
                width: "50%",
                paddingRight: "1rem",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <FinalPrediction predictions={predictions} />
              <PredictionBreakdwon predictions={predictions} />
            </div>
            <div
              style={{ width: "50%", paddingLeft: "1rem", textAlign: "center" }}
            >
              <FullUploadCard
                selectedImage={selectedImage}
                imagePreview={imagePreview}
                onImageSelect={handleImageSelect}
                onClear={handleClear}
                onSubmit={handleSubmit}
                loading={loading}
              />
            </div>
          </div>
        ) : (
          <div
            style={{
              width: "100%",
              paddingLeft: "1rem",
              textAlign: "center",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <FullUploadCard
              selectedImage={selectedImage}
              imagePreview={imagePreview}
              onImageSelect={handleImageSelect}
              onClear={handleClear}
              onSubmit={handleSubmit}
              loading={loading}
            />
          </div>
        )}
      </div>
    </BackgroundContainer>
  );
}
