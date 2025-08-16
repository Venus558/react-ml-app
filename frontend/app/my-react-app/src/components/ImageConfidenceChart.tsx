"use client";

import { Box, Typography } from "@mui/material";
import { BarChart } from "@mui/x-charts/BarChart";

interface Props {
  predictions: { label: string; confidence: number }[];
}

export const ImageConfidenceChart: React.FC<Props> = ({ predictions }) => {
  // Optional: custom colours by label
  const colorMap: Record<string, string> = {
    dog: "#36A2EB",
    cat: "#FF6384",
    other: "#FFCE56",
  };

  const formatted = predictions.map((p) => ({
    label: p.label.charAt(0).toUpperCase() + p.label.slice(1), // Dog, Cat, Other
    value: p.confidence,
    color: colorMap[p.label] || "#888",
  }));

  const seriesData = formatted.map((item, index) => ({
    data: formatted.map((_, i) => (i === index ? item.value : 0)),
    label: item.label,
    color: item.color,
    stack: "total",
  }));

  return (
    <Box sx={{ width: "100%", height: 400, p: 2 }}>
      <Typography variant="h5" align="center" gutterBottom>
        Prediction Confidence
      </Typography>

      <BarChart
        xAxis={[
          {
            scaleType: "band",
            data: formatted.map((item) => item.label),
          },
        ]}
        series={seriesData}
        height={300}
        margin={{ top: 10, right: 30, bottom: 50, left: 50 }}
        slotProps={{
          tooltip: { trigger: "item" },
        }}
      />
    </Box>
  );
};
