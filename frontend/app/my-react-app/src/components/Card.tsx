import type React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = "" }) => (
  <div
    className={className}
    style={{
      backgroundColor: "white",
      borderRadius: "12px",
      boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
      padding: "1.5rem",
      maxWidth: "600px",
      width: "100%",
      boxSizing: "border-box",
    }}
  >
    {children}
  </div>
);
