import React from "react";

interface FinalPredictionProps {
  predictions: { label: string; confidence: number }[];
}

export const FinalPrediction: React.FC<FinalPredictionProps> = ({
  predictions,
}) => {
  if (!predictions || predictions.length === 0) return null;

  const best = predictions.reduce((max, curr) =>
    curr.confidence > max.confidence ? curr : max
  );

  const labelMap: Record<string, string> = {
    cat: "🐱 Cat",
    dog: "🐶 Dog",
    other: "❓ Other",
  };

  return (
    <div
      style={{
        textAlign: "center",
        fontSize: "2.5rem",
        fontWeight: "bold",
        marginBottom: "20px",
        color: "#1a1a1a",
      }}
    >
      Prediction: {labelMap[best.label] || "Unknown"}
    </div>
  );
};
