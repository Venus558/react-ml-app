import React from "react";

interface PredictionBreakdwonProps {
  predictions: { label: string; confidence: number }[];
}

const labelMap: Record<string, { icon: string; name: string; color: string }> =
  {
    cat: { icon: "🐱", name: "Cat", color: "#FF6384" },
    dog: { icon: "🐶", name: "Dog", color: "#36A2EB" },
    other: { icon: "❓", name: "Other", color: "#FFCE56" },
  };

export const PredictionBreakdwon: React.FC<PredictionBreakdwonProps> = ({
  predictions,
}) => {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        marginTop: "20px",
        width: "100%",
      }}
    >
      <div style={{ maxWidth: "400px", width: "100%", textAlign: "left" }}>
        {predictions.map((p) => {
          const { icon, name, color } = labelMap[p.label] || labelMap.other;
          const percent = Math.min(100, Math.max(0, p.confidence));

          return (
            <div key={p.label} style={{ marginBottom: "16px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontWeight: 600,
                  marginBottom: "4px",
                }}
              >
                <span style={{ fontSize: "0.9em", color: "#1a1a1a" }}>
                  {icon} {name}
                </span>
                <span style={{ fontSize: "0.9em", color: "#1a1a1a" }}>
                  {percent.toFixed(1)}%
                </span>
              </div>
              <div
                style={{
                  backgroundColor: "#eee",
                  height: "10px",
                  borderRadius: "5px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${percent}%`,
                    backgroundColor: color,
                    borderRadius: "5px",
                    transition: "width 0.6s ease",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
