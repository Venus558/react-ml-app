import { Brain, TrendingUp } from "lucide-react";

interface SectionBannerProps {
  title: string;
  subtitle?: string;
  icon?: "brain" | "trending";
  variant?: "default" | "accent";
}

export default function SectionBanner({
  title,
  subtitle,
  icon = "brain",
  variant = "default",
}: SectionBannerProps) {
  const IconComponent = icon === "brain" ? Brain : TrendingUp;

  const bannerStyle = {
    width: "100%",
    margin: "2rem auto",
    padding: "1rem 1.5rem",
    borderRadius: "1rem",
    boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
    border: "1px solid rgba(229, 231, 235, 0.5)",
    ...(variant === "accent"
      ? {
          background:
            "linear-gradient(90deg, rgba(239, 246, 255, 1), rgba(245, 243, 255, 1))",
          borderColor: "rgba(147, 197, 253, 0.5)",
        }
      : {
          background: "rgba(255, 255, 255, 0.6)",
          backdropFilter: "blur(8px)",
        }),
  };

  const iconContainerStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "2rem",
    height: "2rem",
    borderRadius: "0.5rem",
    ...(variant === "accent"
      ? {
          background: "linear-gradient(135deg, #3b82f6, #9333ea)",
        }
      : {
          background: "#f3f4f6",
        }),
  };

  const iconStyle = {
    width: "1rem",
    height: "1rem",
    color: variant === "accent" ? "white" : "#4b5563",
  };

  return (
    <div style={bannerStyle}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
        }}
      >
        <div style={iconContainerStyle}>
          <IconComponent style={iconStyle} />
        </div>
        <div>
          <h2
            style={{
              fontSize: window.innerWidth < 640 ? "1.125rem" : "1.25rem",
              fontWeight: "600",
              color: "#111827",
              margin: "0",
            }}
          >
            {title}
          </h2>
          {subtitle && (
            <p
              style={{
                fontSize: "0.875rem",
                color: "#4b5563",
                margin: "0.25rem 0 0 0",
              }}
            >
              {subtitle}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
