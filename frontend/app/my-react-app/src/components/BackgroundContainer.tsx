import React, { useEffect, useState } from "react";

interface BackgroundContainerProps {
  children: React.ReactNode;
}

export const BackgroundContainer: React.FC<BackgroundContainerProps> = ({
  children,
}) => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile(); // run once
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        minHeight: "100dvh",
        background: "linear-gradient(to bottom right, #dbeafe, #e0e7ff)",
        display: isMobile ? "block" : "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "1rem",
        boxSizing: "border-box",
        overflowY: "auto",
      }}
    >
      {children}
    </div>
  );
};
