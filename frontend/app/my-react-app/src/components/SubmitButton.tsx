"use client";

import React, { useState } from "react";

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  variant?: "primary" | "outline";
  loading?: boolean;
}

export const SubmitButton: React.FC<ButtonProps> = ({
  children,
  onClick,
  disabled = false,
  className = "",
  variant = "primary",
  loading = false,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const baseStyle: React.CSSProperties = {
    padding: "0.5rem 1rem",
    borderRadius: "0.5rem",
    fontWeight: 500,
    transition: "all 0.2s ease-in-out",
    opacity: disabled ? 0.5 : 1,
    cursor: disabled ? "not-allowed" : "pointer",
  };

  const variantStyles: Record<string, React.CSSProperties> = {
    primary: {
      backgroundColor: isHovered ? "#3cdf7bff" : "#2563eb", // hover darker blue
      color: "white",
      border: "none",
    },
    outline: {
      backgroundColor: isHovered ? "#f3f4f6" : "transparent", // light gray on hover
      border: "2px solid #a7a7a7ff",
      color: "#818691ff",
    },
  };

  const combinedStyle: React.CSSProperties = {
    ...baseStyle,
    ...variantStyles[variant],
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={className}
      style={combinedStyle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {loading ? (
        <>
          <style>
            {`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}
          </style>
          <div
            style={{
              width: "1.25rem",
              height: "1.25rem",
              border: "2px solid white",
              borderTop: "2px solid transparent",
              borderRadius: "50%",
              animation: "spin 0.6s linear infinite",
              margin: "0 auto",
            }}
          />
        </>
      ) : (
        children
      )}
    </button>
  );
};
