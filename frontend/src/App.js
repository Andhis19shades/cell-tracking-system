import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import "./App.css";


// ============================================================
// API
// ============================================================

const API_BASE = "http://127.0.0.1:5000";

const ANALYSIS_PATH = `${API_BASE}/analysis`;


// ============================================================
// GENERAL HELPERS
// ============================================================

function safeNumber(value) {
  const number = Number(value);

  return Number.isFinite(number)
    ? number
    : 0;
}


function hasNumber(value) {
  const number = Number(value);

  return Number.isFinite(number);
}


function formatNumber(
  value,
  decimals = 2
) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return number.toLocaleString(
    undefined,
    {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }
  );
}


function formatInteger(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return Math.round(number).toLocaleString();
}


function formatDuration(seconds) {
  const number = Number(seconds);

  if (
    !Number.isFinite(number) ||
    number < 0
  ) {
    return "—";
  }

  if (number < 60) {
    return `${Math.round(number)} sec`;
  }

  const minutes = Math.floor(
    number / 60
  );

  const secondsRemaining = Math.round(
    number % 60
  );

  if (minutes < 60) {
    return `${minutes} min ${secondsRemaining} sec`;
  }

  const hours = Math.floor(
    minutes / 60
  );

  const remainingMinutes =
    minutes % 60;

  return `${hours} hr ${remainingMinutes} min`;
}


function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }

  const units = [
    "B",
    "KB",
    "MB",
    "GB",
  ];

  let number = Number(bytes);

  let index = 0;

  while (
    number >= 1024 &&
    index < units.length - 1
  ) {
    number /= 1024;
    index += 1;
  }

  return `${number.toFixed(
    index > 0 ? 2 : 0
  )} ${units[index]}`;
}


function cleanPathPart(value) {
  return String(value || "")
    .replace(/^\/+/, "")
    .replace(/\\/g, "/");
}


function joinUrl(
  base,
  file
) {
  if (!base || !file) {
    return "";
  }

  return `${base}/${cleanPathPart(file)}`;
}


function pickFirst(
  object,
  keys
) {
  if (!object) {
    return null;
  }

  for (const key of keys) {
    if (
      object[key] !== undefined &&
      object[key] !== null &&
      object[key] !== ""
    ) {
      return object[key];
    }
  }

  return null;
}


function findFileByKeywords(
  files,
  keywordGroups
) {
  if (!files) {
    return null;
  }

  const entries = Object.entries(
    files
  );

  for (const group of keywordGroups) {
    const entry = entries.find(
      ([key]) => {
        const lower = key.toLowerCase();

        return group.every(
          (keyword) =>
            lower.includes(
              keyword.toLowerCase()
            )
        );
      }
    );

    if (entry) {
      return entry[1];
    }
  }

  return null;
}


function titleFromKey(key) {
  return String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase()
    );
}


// ============================================================
// CSV PARSER
// ============================================================

function parseCSVLine(line) {
  const result = [];

  let current = "";

  let quoted = false;

  for (
    let index = 0;
    index < line.length;
    index += 1
  ) {
    const character =
      line[index];

    if (character === '"') {
      if (
        quoted &&
        line[index + 1] === '"'
      ) {
        current += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }

      continue;
    }

    if (
      character === "," &&
      !quoted
    ) {
      result.push(current);
      current = "";
      continue;
    }

    current += character;
  }

  result.push(current);

  return result;
}


function parseCSV(text) {
  const lines = String(text || "")
    .replace(/\r/g, "")
    .split("\n")
    .filter(
      (line) => line.trim() !== ""
    );

  if (lines.length < 2) {
    return [];
  }

  const headers = parseCSVLine(
    lines[0]
  ).map(
    (header) =>
      header.trim()
  );

  return lines
    .slice(1)
    .map((line) => {
      const values =
        parseCSVLine(line);

      const row = {};

      headers.forEach(
        (header, index) => {
          row[header] =
            values[index] ??
            "";
        }
      );

      return row;
    });
}


async function loadCSV(
  url
) {
  if (!url) {
    return [];
  }

  const response =
    await fetch(url);

  if (!response.ok) {
    throw new Error(
      `Could not load ${url}`
    );
  }

  return parseCSV(
    await response.text()
  );
}


async function loadCSVOptional(
  url
) {
  if (!url) {
    return [];
  }

  try {
    return await loadCSV(url);
  } catch (
    error
  ) {
    console.warn(
      "Optional CSV unavailable:",
      url,
      error
    );

    return [];
  }
}


// ============================================================
// ICON
// ============================================================

function Icon({
  name,
  size = 18,
}) {
  const paths = {
    upload:
      "M12 16V4m0 0L7 9m5-5l5 5M5 20h14",

    settings:
      "M12 15.5a3.5 3.5 0 100-7 3.5 3.5 0 000 7z M19.4 15a1.8 1.8 0 000 2.5l.1.1-1.8 1.8-.1-.1a1.8 1.8 0 00-2.5 0 1.8 1.8 0 00-.5 1.3v.2h-2.5v-.2a1.8 1.8 0 00-1.8-1.8 1.8 1.8 0 00-1.3.5l-.1.1-1.8-1.8.1-.1a1.8 1.8 0 000-2.5 1.8 1.8 0 00-1.3-.5h-.2V12h.2a1.8 1.8 0 001.8-1.8 1.8 1.8 0 00-.5-1.3l-.1-.1 1.8-1.8.1.1a1.8 1.8 0 002.5 0 1.8 1.8 0 00.5-1.3v-.2h2.5v.2a1.8 1.8 0 001.8 1.8 1.8 1.8 0 001.3-.5l.1-.1 1.8 1.8-.1.1a1.8 1.8 0 000 2.5 1.8 1.8 0 001.3.5h.2v2.5h-.2a1.8 1.8 0 00-1.3.5z",

    chart:
      "M4 19V5m0 14h16M8 16v-4m4 4V8m4 8V5",

    tracking:
      "M4 17c4-8 8 8 16-4M4 17h.01M20 13h.01",

    layers:
      "M12 3l8 4-8 4-8-4 8-4zm-8 8l8 4 8-4M4 15l8 4 8-4",

    download:
      "M12 4v11m0 0l-4-4m4 4l4-4M5 20h14",

    play:
      "M8 5l11 7-11 7V5z",

    search:
      "M11 18a7 7 0 100-14 7 7 0 000 14zm5-2l4 4",

    microscope:
      "M6 18h12M9 18a5 5 0 005-5V8M14 13a4 4 0 10-4-4M12 4h4v4h-4zM17 13h2a3 3 0 013 3v2",

    pulse:
      "M3 12h4l2-6 4 12 2-6h6",

    branch:
      "M6 5v9a4 4 0 004 4h8M6 9h5a3 3 0 003-3V5",

    shield:
      "M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6l7-3z",
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path
        d={
          paths[name] ||
          paths.chart
        }
      />
    </svg>
  );
}


// ============================================================
// UI COMPONENTS
// ============================================================

function SectionHeading({
  number,
  title,
  description,
}) {
  return (
    <div className="section-heading">
      <div>
        <div className="section-number">
          {number}
        </div>

        <h2>{title}</h2>

        {description && (
          <p>
            {description}
          </p>
        )}
      </div>
    </div>
  );
}


function StatCard({
  label,
  value,
  unit,
  detail,
  accent = false,
}) {
  return (
    <div
      className={`stat-card ${
        accent
          ? "stat-accent"
          : ""
      }`}
    >
      <div className="stat-label">
        {label}
      </div>

      <div className="stat-value">
        {value}

        {unit && (
          <span className="stat-unit">
            {unit}
          </span>
        )}
      </div>

      {detail && (
        <div className="stat-detail">
          {detail}
        </div>
      )}
    </div>
  );
}


function InfoItem({
  label,
  value,
}) {
  return (
    <div className="info-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}


function DownloadCard({
  title,
  description,
  url,
  icon = "↓",
}) {
  if (!url) {
    return null;
  }

  return (
    <a
      className="download-card"
      href={url}
      download
    >
      <div className="download-icon">
        {icon}
      </div>

      <div className="download-info">
        <strong>
          {title}
        </strong>

        <span>
          {description}
        </span>
      </div>

      <div className="download-arrow">
        <Icon
          name="download"
          size={18}
        />
      </div>
    </a>
  );
}


function GraphCard({
  title,
  description,
  file,
  baseUrl,
}) {
  if (!file) {
    return null;
  }

  const url =
    joinUrl(
      baseUrl,
      file
    );

  return (
    <div
      className="analysis-graph-card"
    >
      <div className="analysis-graph-header">
        <div>
          <div className="graph-category">
            QUANTITATIVE
          </div>

          <h3>
            {title}
          </h3>

          {description && (
            <p>
              {description}
            </p>
          )}
        </div>

        <a
          className="graph-open"
          href={url}
          target="_blank"
          rel="noreferrer"
        >
          ↗
        </a>
      </div>

      <div className="analysis-graph-image">
        <img
          src={url}
          alt={title}
        />
      </div>
    </div>
  );
}


function ProgressPanel({
  progress,
  message,
}) {
  const percent = Math.max(
    0,
    Math.min(
      100,
      safeNumber(
        progress.percent
      )
    )
  );

  return (
    <div
      className="loading-card"
      style={{
        display: "block",
        padding: 24,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems:
            "center",
          gap: 20,
          marginBottom: 14,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems:
              "center",
            gap: 12,
          }}
        >
          <span className="loading-spinner" />

          <div>
            <strong
              style={{
                display: "block",
                fontSize: 16,
                marginBottom: 4,
              }}
            >
              ANALYSIS IN PROGRESS
            </strong>

            <span
              style={{
                opacity: 0.72,
              }}
            >
              {message ||
                "Processing microscopy data…"}
            </span>
          </div>
        </div>

        <strong
          style={{
            fontSize: 30,
            whiteSpace:
              "nowrap",
          }}
        >
          {Math.round(
            percent
          )}
          %
        </strong>
      </div>

      <div
        style={{
          height: 10,
          width: "100%",
          background:
            "rgba(255,255,255,0.08)",
          borderRadius: 999,
          overflow: "hidden",
          marginBottom: 18,
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${percent}%`,
            background:
              "linear-gradient(90deg, #36d9b6, #7ac7ff)",
            borderRadius: 999,
            transition:
              "width .35s ease",
          }}
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(4,minmax(0,1fr))",
          gap: 12,
        }}
      >
        <div className="progress-stat">
          <span>FRAME</span>

          <strong>
            {formatInteger(
              progress.currentFrame
            )}
            {" / "}
            {formatInteger(
              progress.totalFrames
            )}
          </strong>
        </div>

        <div className="progress-stat">
          <span>
            PROCESSING SPEED
          </span>

          <strong>
            {formatNumber(
              progress.framesPerSecond,
              2
            )}
            {" "}
            frames/s
          </strong>
        </div>

        <div className="progress-stat">
          <span>ELAPSED</span>

          <strong>
            {formatDuration(
              progress.elapsedSeconds
            )}
          </strong>
        </div>

        <div className="progress-stat">
          <span>
            EST. TIME LEFT
          </span>

          <strong>
            {progress.etaSeconds ===
            null
              ? "Calculating…"
              : formatDuration(
                  progress.etaSeconds
                )}
          </strong>
        </div>
      </div>
    </div>
  );
}


// ============================================================
// MICROGLIA INTERPRETATION
// ============================================================

function MicrogliaInterpretation({
  microglia,
  population,
  experiment,
}) {
  const extensions =
    safeNumber(
      microglia?.process_extension_events ??
        microglia?.processExtensions
    );

  const retractions =
    safeNumber(
      microglia?.process_retraction_events ??
        microglia?.processRetractions
    );

  const totalTipEvents =
    extensions +
    retractions;

  const extensionFraction =
    totalTipEvents > 0
      ? extensions /
        totalTipEvents
      : 0;

  const balanceLabel =
    totalTipEvents === 0
      ? "No process-tip transitions"
      : extensionFraction >
          0.55
        ? "Extension-dominant"
        : extensionFraction <
            0.45
          ? "Retraction-dominant"
          : "Balanced dynamics";

  return (
    <div
      style={{
        border:
          "1px solid rgba(100,181,255,.18)",
        background:
          "linear-gradient(145deg, rgba(19,31,45,.95), rgba(12,24,32,.96))",
        borderRadius: 14,
        padding: 22,
        marginBottom: 24,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems:
            "flex-start",
          gap: 20,
          marginBottom: 22,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 10,
              letterSpacing: ".16em",
              color: "#6bd8c6",
              marginBottom: 8,
            }}
          >
            SCIENTIFIC INTERPRETATION
          </div>

          <h3
            style={{
              margin: 0,
              fontSize: 22,
            }}
          >
            Microglial morphology &
            motility
          </h3>

          <p
            style={{
              margin:
                "8px 0 0",
              maxWidth: 760,
              opacity: 0.72,
              lineHeight: 1.6,
            }}
          >
            The dashboard combines AI-based
            segmentation with instance
            separation, skeleton topology,
            Sholl analysis and temporal
            process-tip tracking.
          </p>
        </div>

        <div
          style={{
            padding:
              "8px 12px",
            border:
              "1px solid rgba(83,214,194,.25)",
            borderRadius: 999,
            fontSize: 10,
            letterSpacing: ".1em",
            color: "#7de5d3",
            whiteSpace:
              "nowrap",
          }}
        >
          MICROGLIA MODE
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(4,minmax(0,1fr))",
          gap: 12,
          marginBottom: 20,
        }}
      >
        <InterpretationMetric
          label="ARBOR LENGTH"
          value={formatNumber(
            microglia?.mean_skeleton_length_um
          )}
          unit="µm"
          icon="branch"
        />

        <InterpretationMetric
          label="BRANCHING"
          value={formatNumber(
            microglia?.mean_branch_points
          )}
          unit="junctions"
          icon="branch"
        />

        <InterpretationMetric
          label="SHOLL MAX"
          value={formatNumber(
            microglia?.mean_sholl_max_intersections
          )}
          unit="intersections"
          icon="chart"
        />

        <InterpretationMetric
          label="TIP DYNAMICS"
          value={balanceLabel}
          unit=""
          icon="pulse"
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(2,minmax(0,1fr))",
          gap: 16,
        }}
      >
        <InterpretationBlock
          title="Morphology"
          text={`Mean projected cell area is ${formatNumber(
            microglia?.mean_cell_area_um2
          )} µm² with an average soma-area proxy of ${formatNumber(
            microglia?.mean_soma_area_proxy_um2
          )} µm². Arbor complexity is represented by skeleton length, endpoints, branch points and Sholl intersections.`}
        />

        <InterpretationBlock
          title="Motility"
          text={`The system measures soma displacement separately from process-tip displacement. Current tip transitions contain ${formatInteger(
            extensions
          )} extension events and ${formatInteger(
            retractions
          )} retraction events.`}
        />

        <InterpretationBlock
          title="Population"
          text={`Detected cell count ranges from ${formatInteger(
            population?.minimum_cell_count ??
              population?.minimumCells
          )} to ${formatInteger(
            population?.maximum_cell_count ??
              population?.maximumCells
          )}. Population change is an imaging-derived estimate rather than a definitive proliferation measurement.`}
        />

        <InterpretationBlock
          title="Methodology distinction"
          text="This implementation does more than centroid-only tracking: the microglia route combines semantic U-Net segmentation, distance-transform/watershed instance separation, soma proxies, skeleton topology, Sholl analysis and process-tip dynamics in one workflow."
        />
      </div>

      <div
        style={{
          marginTop: 18,
          paddingTop: 16,
          borderTop:
            "1px solid rgba(255,255,255,.08)",
          fontSize: 11,
          lineHeight: 1.7,
          opacity: 0.64,
        }}
      >
        <strong>
          Interpretation note:
        </strong>{" "}
        morphology-derived measurements
        describe image structure and movement.
        They do not by themselves establish
        activation state, apoptosis,
        metabolism or another biological state.
        Accurate µm/min values require the
        real acquisition interval.
        {experiment?.pixel_calibration_um_per_px
          ? ` Current calibration: ${formatNumber(
              experiment.pixel_calibration_um_per_px,
              3
            )} µm/px.`
          : ""}
      </div>
    </div>
  );
}


function InterpretationMetric({
  label,
  value,
  unit,
  icon,
}) {
  return (
    <div
      style={{
        border:
          "1px solid rgba(255,255,255,.07)",
        borderRadius: 10,
        padding: 14,
        background:
          "rgba(255,255,255,.025)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems:
            "center",
          gap: 7,
          fontSize: 10,
          letterSpacing: ".1em",
          opacity: 0.58,
          marginBottom: 8,
        }}
      >
        <Icon
          name={icon}
          size={14}
        />

        {label}
      </div>

      <div
        style={{
          fontSize: 20,
          fontWeight: 600,
        }}
      >
        {value}

        {unit && (
          <span
            style={{
              fontSize: 11,
              marginLeft: 6,
              opacity: 0.55,
            }}
          >
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}


function InterpretationBlock({
  title,
  text,
}) {
  return (
    <div
      style={{
        border:
          "1px solid rgba(255,255,255,.06)",
        borderRadius: 10,
        padding: 16,
        background:
          "rgba(255,255,255,.018)",
      }}
    >
      <div
        style={{
          fontSize: 11,
          letterSpacing: ".08em",
          color: "#76d8cb",
          marginBottom: 8,
        }}
      >
        {title.toUpperCase()}
      </div>

      <div
        style={{
          fontSize: 12,
          lineHeight: 1.7,
          opacity: 0.78,
        }}
      >
        {text}
      </div>
    </div>
  );
}


// ============================================================
// APP
// ============================================================

function App() {
  const [files, setFiles] =
    useState([]);

  const [analysisMode, setAnalysisMode] =
    useState("microglia");

  const [settings, setSettings] =
    useState({
      fps: 8,
      pixelSize: "0.325",
      frameInterval: "",
      processEveryNFrames: 4,
      inferenceBatchSize: 8,
    });

  const [frameData, setFrameData] =
    useState([]);

  const [trackData, setTrackData] =
    useState([]);

  const [summaryData, setSummaryData] =
    useState([]);

  const [tipData, setTipData] =
    useState([]);

  const [populationData, setPopulationData] =
    useState([]);

  const [resultMeta, setResultMeta] =
    useState(null);

  const [jobId, setJobId] =
    useState(null);

  const [jobStatus, setJobStatus] =
    useState("idle");

  const [progressMessage, setProgressMessage] =
    useState("");

  const [analysisProgress, setAnalysisProgress] =
    useState({
      percent: 0,
      currentFrame: 0,
      totalFrames: 0,
      elapsedSeconds: 0,
      framesPerSecond: 0,
      etaSeconds: null,
    });

  const [dataError, setDataError] =
    useState("");

  const [selectedCell, setSelectedCell] =
    useState(null);

  const [search, setSearch] =
    useState("");

  const [sortBy, setSortBy] =
    useState(
      "mean_soma_speed_um_min"
    );

  const [activeGraphCategory, setActiveGraphCategory] =
    useState("All");

  const [mobileMenu, setMobileMenu] =
    useState(false);

  const [backendOnline, setBackendOnline] =
    useState(false);

  const [microgliaInfo, setMicrogliaInfo] =
    useState(null);

  const objectUrlRef =
    useRef(null);


  // ==========================================================
  // BACKEND STATUS
  // ==========================================================

  useEffect(() => {
    let cancelled = false;

    const checkBackend =
      async () => {
        try {
          const response =
            await fetch(
              `${API_BASE}/api/health`
            );

          if (!cancelled) {
            setBackendOnline(
              response.ok
            );
          }
        } catch (
          error
        ) {
          if (!cancelled) {
            setBackendOnline(
              false
            );
          }
        }
      };

    const checkMicroglia =
      async () => {
        try {
          const response =
            await fetch(
              `${API_BASE}/api/microglia/info`
            );

          if (!response.ok) {
            return;
          }

          const data =
            await response.json();

          if (!cancelled) {
            setMicrogliaInfo(
              data
            );
          }
        } catch (
          error
        ) {
          console.warn(
            "Microglia info unavailable:",
            error
          );
        }
      };

    checkBackend();

    checkMicroglia();

    const timer =
      setInterval(
        checkBackend,
        5000
      );

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);


  // ==========================================================
  // CLEANUP
  // ==========================================================

  useEffect(() => {
    return () => {
      if (
        objectUrlRef.current
      ) {
        URL.revokeObjectURL(
          objectUrlRef.current
        );
      }
    };
  }, []);


  // ==========================================================
  // FILE SELECTION
  // ==========================================================

  const isSupportedFile =
    (file) => {
      if (!file) {
        return false;
      }

      return (
        file.type.startsWith(
          "video/"
        ) ||
        file.type ===
          "image/tiff" ||
        /\.(avi|mp4|mov|mkv|webm|mpg|mpeg|tif|tiff|png|jpg|jpeg)$/i.test(
          file.name
        )
      );
    };


  const selectFiles = (
    selected
  ) => {
    const file =
      Array.from(
        selected || []
      ).find(
        isSupportedFile
      );

    if (!file) {
      setDataError(
        "Please select a supported microscopy file: AVI, MP4, MOV, MKV, WEBM, TIFF, PNG or JPEG."
      );

      return;
    }

    if (
      objectUrlRef.current
    ) {
      URL.revokeObjectURL(
        objectUrlRef.current
      );
    }

    objectUrlRef.current =
      URL.createObjectURL(
        file
      );

    setFiles([
      file,
    ]);

    setDataError("");

    setJobStatus(
      "selected"
    );

    setProgressMessage(
      `${file.name} selected — ready for analysis.`
    );

    setAnalysisProgress({
      percent: 0,
      currentFrame: 0,
      totalFrames: 0,
      elapsedSeconds: 0,
      framesPerSecond: 0,
      etaSeconds: null,
    });

    setFrameData([]);
    setTrackData([]);
    setSummaryData([]);
    setTipData([]);
    setPopulationData([]);
    setResultMeta(null);
    setSelectedCell(null);
    setJobId(null);
  };


  const handleFiles = (
    event
  ) => {
    selectFiles(
      event.target.files
    );

    event.target.value =
      "";
  };


  const handleDrop = (
    event
  ) => {
    event.preventDefault();

    selectFiles(
      event.dataTransfer.files
    );
  };


  const clearFiles = () => {
    if (
      objectUrlRef.current
    ) {
      URL.revokeObjectURL(
        objectUrlRef.current
      );

      objectUrlRef.current =
        null;
    }

    setFiles([]);
    setJobId(null);
    setJobStatus("idle");
    setProgressMessage("");

    setAnalysisProgress({
      percent: 0,
      currentFrame: 0,
      totalFrames: 0,
      elapsedSeconds: 0,
      framesPerSecond: 0,
      etaSeconds: null,
    });

    setFrameData([]);
    setTrackData([]);
    setSummaryData([]);
    setTipData([]);
    setPopulationData([]);
    setResultMeta(null);
    setSelectedCell(null);
    setDataError("");
  };


  // ==========================================================
  // SETTINGS
  // ==========================================================

  const updateSetting = (
    name,
    value
  ) => {
    setSettings(
      (previous) => ({
        ...previous,
        [name]: value,
      })
    );
  };


  useEffect(() => {
    if (
      analysisMode ===
      "microglia"
    ) {
      setSettings(
        (previous) => ({
          ...previous,
          pixelSize:
            previous.pixelSize ||
            "0.325",
          processEveryNFrames: 4,
          inferenceBatchSize: 8,
        })
      );

      return;
    }

    setSettings(
      (previous) => ({
        ...previous,
        processEveryNFrames:
          previous.processEveryNFrames ||
          2,
        inferenceBatchSize:
          previous.inferenceBatchSize ||
          4,
      })
    );
  }, [
    analysisMode,
  ]);


  // ==========================================================
  // ANALYSIS
  // ==========================================================

  const analyzeVideo =
    async () => {
      const file =
        files[0];

      if (!file) {
        setDataError(
          "Select a microscopy file first."
        );

        return;
      }

      setDataError("");

      setJobStatus(
        "uploading"
      );

      setProgressMessage(
        `Uploading ${file.name} to the local AI engine…`
      );

      setFrameData([]);
      setTrackData([]);
      setSummaryData([]);
      setTipData([]);
      setPopulationData([]);
      setResultMeta(null);
      setSelectedCell(null);

      setAnalysisProgress({
        percent: 0,
        currentFrame: 0,
        totalFrames: 0,
        elapsedSeconds: 0,
        framesPerSecond: 0,
        etaSeconds: null,
      });

      const formData =
        new FormData();

      formData.append(
        "video",
        file
      );

      formData.append(
        "analysis_mode",
        analysisMode
      );

      formData.append(
        "process_every_n_frames",
        String(
          Math.max(
            1,
            Math.min(
              10,
              Number(
                settings.processEveryNFrames
              ) || 1
            )
          )
        )
      );

      formData.append(
        "inference_batch_size",
        String(
          Math.max(
            1,
            Math.min(
              8,
              Number(
                settings.inferenceBatchSize
              ) || 1
            )
          )
        )
      );

      if (
        settings.pixelSize
      ) {
        formData.append(
          "pixel_size_um",
          String(
            settings.pixelSize
          )
        );
      }

      if (
        analysisMode ===
          "microglia" &&
        settings.frameInterval
      ) {
        formData.append(
          "frame_interval_seconds",
          String(
            settings.frameInterval
          )
        );
      }

      try {
        const response =
          await fetch(
            `${API_BASE}/api/analyze`,
            {
              method: "POST",
              body: formData,
            }
          );

        const body =
          await response
            .json()
            .catch(
              () => ({})
            );

        if (!response.ok) {
          throw new Error(
            body.error ||
              body.message ||
              `Analysis request failed (${response.status})`
          );
        }

        const id =
          body.job_id ||
          body.jobId ||
          body.id;

        if (!id) {
          throw new Error(
            "Backend did not return a job ID."
          );
        }

        setJobId(id);

        setJobStatus(
          "processing"
        );

        setProgressMessage(
          analysisMode ===
            "microglia"
            ? `Analyzing ${file.name} with the Microglia AI pipeline…`
            : `Analyzing ${file.name} with the Cell Lab pipeline…`
        );

        document
          .getElementById(
            "results"
          )
          ?.scrollIntoView({
            behavior:
              "smooth",
            block: "start",
          });
      } catch (
        error
      ) {
        console.error(
          "Analysis error:",
          error
        );

        setJobStatus(
          "error"
        );

        setDataError(
          error.message ||
            "Could not start analysis."
        );

        setProgressMessage("");
      }
    };


  // ==========================================================
  // RESULTS LOADING
  // ==========================================================

  const loadResults =
    async (
      id
    ) => {
      const base =
        `${ANALYSIS_PATH}/${id}`;

      const resultResponse =
        await fetch(
          `${API_BASE}/api/results/${id}`
        );

      if (
        !resultResponse.ok
      ) {
        throw new Error(
          `Could not load analysis result (${resultResponse.status}).`
        );
      }

      const result =
        await resultResponse.json();

      setResultMeta(
        result
      );

      const filesMap =
        result.files ||
        {};

      const frameFile =
        findFileByKeywords(
          filesMap,
          [
            [
              "frame",
              "metrics",
            ],
            [
              "frame",
              "analysis",
            ],
            [
              "frame",
            ],
          ]
        );

      const trackFile =
        findFileByKeywords(
          filesMap,
          [
            [
              "cell",
              "tracks",
            ],
            [
              "cell",
              "motion",
            ],
            [
              "tracks",
            ],
          ]
        );

      const summaryFile =
        findFileByKeywords(
          filesMap,
          [
            [
              "track",
              "summary",
            ],
            [
              "cell",
              "summary",
            ],
            [
              "summary",
            ],
          ]
        );

      const tipFile =
        findFileByKeywords(
          filesMap,
          [
            [
              "process",
              "tips",
            ],
            [
              "process",
              "tip",
            ],
            [
              "tip",
            ],
          ]
        );

      const populationFile =
        findFileByKeywords(
          filesMap,
          [
            [
              "population",
            ],
          ]
        );

      const [
        frames,
        tracks,
        summaries,
        tips,
        population,
      ] =
        await Promise.all(
          [
            loadCSVOptional(
              frameFile
                ? joinUrl(
                    base,
                    frameFile
                  )
                : ""
            ),

            loadCSVOptional(
              trackFile
                ? joinUrl(
                    base,
                    trackFile
                  )
                : ""
            ),

            loadCSVOptional(
              summaryFile
                ? joinUrl(
                    base,
                    summaryFile
                  )
                : ""
            ),

            loadCSVOptional(
              tipFile
                ? joinUrl(
                    base,
                    tipFile
                  )
                : ""
            ),

            loadCSVOptional(
              populationFile
                ? joinUrl(
                    base,
                    populationFile
                  )
                : ""
            ),
          ]
        );

      setFrameData(
        frames
      );

      setTrackData(
        tracks
      );

      setSummaryData(
        summaries
      );

      setTipData(
        tips
      );

      setPopulationData(
        population
      );
    };


  // ==========================================================
  // STATUS POLLING
  // ==========================================================

  useEffect(() => {
    if (
      !jobId ||
      jobStatus !==
        "processing"
    ) {
      return undefined;
    }

    let cancelled =
      false;

    const poll =
      async () => {
        try {
          const response =
            await fetch(
              `${API_BASE}/api/status/${jobId}`
            );

          const body =
            await response.json();

          if (!response.ok) {
            throw new Error(
              body.error ||
                body.message ||
                "Could not read analysis status."
            );
          }

          if (
            cancelled
          ) {
            return;
          }

          const status =
            String(
              body.status ||
                "processing"
            ).toLowerCase();

          const percent =
            Math.max(
              0,
              Math.min(
                100,
                Number(
                  body.progress ??
                    body.percent ??
                    0
                ) || 0
              )
            );

          const currentFrame =
            Number(
              body.current_frame ??
                body.frame ??
                0
            ) || 0;

          const totalFrames =
            Number(
              body.total_frames ??
                body.total ??
                0
            ) || 0;

          const elapsedSeconds =
            Number(
              body.elapsed_seconds ??
                body.elapsed ??
                0
            ) || 0;

          const framesPerSecond =
            Number(
              body.frames_per_second ??
                body.processing_fps ??
                0
            ) || 0;

          const etaRaw =
            body.eta_seconds ??
            body.eta ??
            null;

          const etaSeconds =
            etaRaw === null ||
            etaRaw === undefined ||
            etaRaw === ""
              ? null
              : Number(
                  etaRaw
                );

          setAnalysisProgress({
            percent,
            currentFrame,
            totalFrames,
            elapsedSeconds,
            framesPerSecond,
            etaSeconds:
              Number.isFinite(
                etaSeconds
              )
                ? etaSeconds
                : null,
          });

          setProgressMessage(
            body.message ||
              body.progress_message ||
              body.detail ||
              "Processing microscopy data…"
          );

          if (
            [
              "complete",
              "completed",
              "done",
              "success",
            ].includes(
              status
            )
          ) {
            setAnalysisProgress(
              (previous) => ({
                ...previous,
                percent: 100,
                currentFrame:
                  totalFrames ||
                  previous.currentFrame,
                etaSeconds: 0,
              })
            );

            setJobStatus(
              "completed"
            );

            setProgressMessage(
              "Analysis complete. Quantitative results are ready."
            );

            try {
              await loadResults(
                jobId
              );
            } catch (
              error
            ) {
              console.error(
                "Could not load result files:",
                error
              );

              setDataError(
                error.message ||
                  "Analysis completed, but result files could not be loaded."
              );
            }
          }

          if (
            [
              "failed",
              "error",
            ].includes(
              status
            )
          ) {
            setJobStatus(
              "error"
            );

            setDataError(
              body.error ||
                body.message ||
                "The analysis failed."
            );
          }
        } catch (
          error
        ) {
          if (
            !cancelled
          ) {
            setJobStatus(
              "error"
            );

            setDataError(
              error.message ||
                "Lost connection to the Python backend."
            );
          }
        }
      };

    poll();

    const timer =
      setInterval(
        poll,
        800
      );

    return () => {
      cancelled = true;
      clearInterval(
        timer
      );
    };
  }, [
    jobId,
    jobStatus,
  ]);


  // ==========================================================
  // RESULT META
  // ==========================================================

  const experiment =
    resultMeta?.experiment ||
    {};

  const backendResultFiles =
    resultMeta?.files ||
    {};

  const microgliaSummary =
    resultMeta?.microglia ||
    {};

  const populationSummary =
    resultMeta?.population_dynamics ||
    {};


  const isCompleted =
    jobStatus ===
    "completed";


  const resultBase =
    jobId
      ? `${ANALYSIS_PATH}/${jobId}`
      : "";


  // ==========================================================
  // FILE OUTPUTS — DYNAMIC
  // ==========================================================

  const annotatedVideoFile =
    pickFirst(
      backendResultFiles,
      [
        "annotated_video",
        "analyzed_video",
        "quantitative_video",
        "video",
      ]
    ) ||
    findFileByKeywords(
      backendResultFiles,
      [
        [
          "annotated",
          "video",
        ],
        [
          "quantitative",
          "video",
        ],
        [
          "video",
        ],
      ]
    );


  const reportFile =
    pickFirst(
      backendResultFiles,
      [
        "report",
        "quantitative_report",
        "analysis_report",
      ]
    ) ||
    findFileByKeywords(
      backendResultFiles,
      [
        [
          "report",
        ],
      ]
    );


  const excelFile =
    pickFirst(
      backendResultFiles,
      [
        "excel",
        "xlsx",
      ]
    ) ||
    findFileByKeywords(
      backendResultFiles,
      [
        [
          "excel",
        ],
        [
          "xlsx",
        ],
      ]
    );


  const annotatedVideoUrl =
    resultBase &&
    annotatedVideoFile
      ? joinUrl(
          resultBase,
          annotatedVideoFile
        )
      : "";


  const reportUrl =
    resultBase &&
    reportFile
      ? joinUrl(
          resultBase,
          reportFile
        )
      : "";


  const excelUrl =
    resultBase &&
    excelFile
      ? joinUrl(
          resultBase,
          excelFile
        )
      : "";


  // ==========================================================
  // GRAPH LIST — DYNAMIC
  // ==========================================================

  const graphFiles =
    resultMeta?.graphs &&
    typeof resultMeta.graphs ===
      "object"
      ? Object.entries(
          resultMeta.graphs
        )
      : [];

  const graphObjects =
    graphFiles.map(
      ([key, value]) => ({
        key,
        title:
          titleFromKey(
            key
          ),
        file: value,
        description:
          graphDescription(
            key,
            analysisMode
          ),
        category:
          graphCategory(
            key
          ),
      })
    );


  const graphCategories = [
    "All",
    ...Array.from(
      new Set(
        graphObjects.map(
          (graph) =>
            graph.category
        )
      )
    ),
  ];


  const filteredGraphs =
    activeGraphCategory ===
    "All"
      ? graphObjects
      : graphObjects.filter(
          (graph) =>
            graph.category ===
            activeGraphCategory
        );


  // ==========================================================
  // METADATA
  // ==========================================================

  const videoName =
    files[0]?.name ||
    experiment.filename ||
    resultMeta?.filename ||
    "No file selected";


  const frameCount =
    experiment.total_frames ??
    experiment.frames ??
    experiment.frame_count ??
    analysisProgress.totalFrames ??
    frameData.length;


  const actualFps =
    experiment.fps ??
    experiment.frame_rate ??
    experiment.video_fps ??
    settings.fps;


  const durationSeconds =
    experiment.duration_seconds ??
    experiment.duration ??
    (
      Number(frameCount) > 0 &&
      Number(actualFps) > 0
        ? Number(frameCount) /
          Number(actualFps)
        : 0
    );


  const width =
    experiment.width ??
    experiment.video_width ??
    "—";


  const height =
    experiment.height ??
    experiment.video_height ??
    "—";


  const pixelSize =
    experiment.pixel_calibration_um_per_px ??
    experiment.pixel_size_um ??
    settings.pixelSize;


  const processStep =
    experiment.analysis_frame_step ??
    settings.processEveryNFrames;


  const inferenceBatch =
    experiment.inference_batch_size ??
    settings.inferenceBatchSize;


  // ==========================================================
  // BACTERIA STATISTICS
  // ==========================================================

  const bacteriaStatistics =
    useMemo(() => {
      if (
        !frameData.length
      ) {
        return {
          averageCells: 0,
          maximumCells: 0,
          averageMoving: 0,
          meanSpeed: 0,
          medianSpeed: 0,
          maximumSpeed: 0,
          meanDisplacement: 0,
          totalDisplacement: 0,
          meanArea: 0,
        };
      }

      const average = (
        values
      ) => {
        if (
          !values.length
        ) {
          return 0;
        }

        return (
          values.reduce(
            (
              sum,
              value
            ) =>
              sum +
              value,
            0
          ) /
          values.length
        );
      };

      const median = (
        values
      ) => {
        if (
          !values.length
        ) {
          return 0;
        }

        const sorted =
          [
            ...values,
          ].sort(
            (a, b) =>
              a - b
          );

        const middle =
          Math.floor(
            sorted.length /
              2
          );

        if (
          sorted.length %
            2
        ) {
          return sorted[
            middle
          ];
        }

        return (
          sorted[
            middle - 1
          ] +
          sorted[
            middle
          ]
        ) / 2;
      };

      const cells =
        frameData.map(
          (row) =>
            safeNumber(
              row.cell_count
            )
        );

      const moving =
        frameData.map(
          (row) =>
            safeNumber(
              row.moving_cells
            )
        );

      const meanSpeeds =
        frameData
          .map(
            (row) =>
              safeNumber(
                row.mean_speed_px_s
              )
          )
          .filter(
            (value) =>
              value > 0
          );

      const medianSpeeds =
        frameData
          .map(
            (row) =>
              safeNumber(
                row.median_speed_px_s
              )
          )
          .filter(
            (value) =>
              value > 0
          );

      const maximumSpeeds =
        frameData.map(
          (row) =>
            safeNumber(
              row.max_speed_px_s
            )
        );

      const displacements =
        frameData
          .map(
            (row) =>
              safeNumber(
                row.mean_displacement_px
              )
          )
          .filter(
            (value) =>
              value > 0
          );

      const totals =
        frameData.map(
          (row) =>
            safeNumber(
              row.total_displacement_px
            )
        );

      const areas =
        frameData
          .map(
            (row) =>
              safeNumber(
                row.mean_cell_area_px
              )
          )
          .filter(
            (value) =>
              value > 0
          );

      return {
        averageCells:
          average(cells),

        maximumCells:
          cells.length
            ? Math.max(
                ...cells
              )
            : 0,

        averageMoving:
          average(moving),

        meanSpeed:
          average(
            meanSpeeds
          ),

        medianSpeed:
          median(
            medianSpeeds
          ),

        maximumSpeed:
          maximumSpeeds.length
            ? Math.max(
                ...maximumSpeeds
              )
            : 0,

        meanDisplacement:
          average(
            displacements
          ),

        totalDisplacement:
          totals.reduce(
            (
              sum,
              value
            ) =>
              sum +
              value,
            0
          ),

        meanArea:
          average(areas),
      };
    }, [
      frameData,
    ]);


  // ==========================================================
  // MICROGLIA STATISTICS
  // ==========================================================

  const microgliaStats =
    useMemo(() => {
      const summary =
        microgliaSummary ||
        {};

      return {
        cellsTracked:
          safeNumber(
            summary.cells_tracked ??
              summary.trackedObjects ??
              summary.tracked_objects
          ),

        cellObservations:
          safeNumber(
            summary.cell_observations
          ),

        meanArea:
          safeNumber(
            summary.mean_cell_area_um2 ??
              summary.meanArea
          ),

        meanSomaArea:
          safeNumber(
            summary.mean_soma_area_proxy_um2
          ),

        meanProcessLength:
          safeNumber(
            summary.mean_skeleton_length_um ??
              summary.meanProcessLength
          ),

        meanBranchPoints:
          safeNumber(
            summary.mean_branch_points ??
              summary.meanBranchPoints
          ),

        meanEndpoints:
          safeNumber(
            summary.mean_endpoints ??
              summary.meanEndpoints
          ),

        meanSholl:
          safeNumber(
            summary.mean_sholl_max_intersections ??
              summary.meanShollMaximum
          ),

        meanSomaSpeed:
          summary.mean_soma_speed_um_min !==
          null &&
          summary.mean_soma_speed_um_min !==
          undefined
            ? Number(
                summary.mean_soma_speed_um_min
              )
            : null,

        meanTipSpeed:
          summary.mean_process_tip_speed_um_min !==
          null &&
          summary.mean_process_tip_speed_um_min !==
          undefined
            ? Number(
                summary.mean_process_tip_speed_um_min
              )
            : null,

        extensions:
          safeNumber(
            summary.process_extension_events ??
              summary.processExtensions
          ),

        retractions:
          safeNumber(
            summary.process_retraction_events ??
              summary.processRetractions
          ),

        disappearanceCandidates:
          safeNumber(
            summary.disappearance_like_candidates
          ),
      };
    }, [
      microgliaSummary,
    ]);


  // ==========================================================
  // TRACK SUMMARY
  // ==========================================================

  const trackCount =
    summaryData.length;


  const microgliaTrackSortField =
    sortBy;


  const filteredSummary =
    useMemo(() => {
      let data =
        [
          ...summaryData,
        ];

      const query =
        search.trim()
          .toLowerCase();

      if (query) {
        data =
          data.filter(
            (row) =>
              String(
                row.track_id ??
                  row.cell_id ??
                  ""
              )
                .toLowerCase()
                .includes(
                  query
                )
          );
      }

      data.sort(
        (a, b) =>
          safeNumber(
            b[
              microgliaTrackSortField
            ]
          ) -
          safeNumber(
            a[
              microgliaTrackSortField
            ]
          )
      );

      return data;
    }, [
      summaryData,
      search,
      microgliaTrackSortField,
    ]);


  // ==========================================================
  // SELECTED TRACK
  // ==========================================================

  const selectedTrack =
    useMemo(() => {
      if (
        !selectedCell
      ) {
        return [];
      }

      const trackId =
        selectedCell.track_id ??
        selectedCell.cell_id;

      return trackData
        .filter(
          (row) =>
            String(
              row.track_id ??
                row.cell_id
            ) ===
            String(
              trackId
            )
        )
        .sort(
          (a, b) =>
            safeNumber(
              a.frame
            ) -
            safeNumber(
              b.frame
            )
        );
    }, [
      selectedCell,
      trackData,
    ]);


  // ==========================================================
  // SELECTED MICROGLIA TIPS
  // ==========================================================

  const selectedTipRows =
    useMemo(() => {
      if (
        !selectedCell ||
        !tipData.length
      ) {
        return [];
      }

      const trackId =
        selectedCell.track_id ??
        selectedCell.cell_id;

      return tipData.filter(
        (row) =>
          String(
            row.track_id ??
              row.cell_id
          ) ===
          String(
            trackId
          )
      );
    }, [
      selectedCell,
      tipData,
    ]);


  // ==========================================================
  // POPULATION
  // ==========================================================

  const populationRows =
    populationData;


  const populationAverage =
    safeNumber(
      populationSummary.average_cell_count ??
        populationSummary.averageCells
    );


  const populationMaximum =
    safeNumber(
      populationSummary.maximum_cell_count ??
        populationSummary.maximumCells
    );


  const populationMinimum =
    safeNumber(
      populationSummary.minimum_cell_count ??
        populationSummary.minimumCells
    );


  // ==========================================================
  // FILE DOWNLOADS
  // ==========================================================

  const downloadEntries =
    useMemo(() => {
      if (
        !resultBase ||
        !backendResultFiles
      ) {
        return [];
      }

      return Object.entries(
        backendResultFiles
      )
        .filter(
          ([, value]) =>
            typeof value ===
              "string" &&
            value.trim() !== ""
        )
        .map(
          ([
            key,
            value,
          ]) => ({
            key,
            title:
              titleFromKey(
                key
              ),
            value,
            url:
              joinUrl(
                resultBase,
                value
              ),
          })
        );
    }, [
      backendResultFiles,
      resultBase,
    ]);


  // ==========================================================
  // DYNAMIC TRACKING SORT OPTIONS
  // ==========================================================

  const trackingSortOptions =
    analysisMode ===
    "microglia"
      ? [
          {
            value:
              "mean_soma_speed_um_min",
            label:
              "Soma speed",
          },
          {
            value:
              "mean_process_tip_speed_um_min",
            label:
              "Process-tip speed",
          },
          {
            value:
              "mean_skeleton_length_um",
            label:
              "Process length",
          },
          {
            value:
              "mean_branch_points",
            label:
              "Branch points",
          },
          {
            value:
              "frames_tracked",
            label:
              "Frames tracked",
          },
        ]
      : [
          {
            value:
              "average_speed_px_s",
            label:
              "Average speed",
          },
          {
            value:
              "maximum_speed_px_s",
            label:
              "Maximum speed",
          },
          {
            value:
              "total_distance_px",
            label:
              "Total distance",
          },
          {
            value:
              "frames_tracked",
            label:
              "Frames tracked",
          },
          {
            value:
              "average_area_px",
            label:
              "Average area",
          },
        ];


  // ==========================================================
  // NAVIGATION
  // ==========================================================

  const scrollToSection =
    (id) => {
      document
        .getElementById(id)
        ?.scrollIntoView({
          behavior:
            "smooth",
          block:
            "start",
        });

      setMobileMenu(
        false
      );
    };


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <>
      <section className="landing-hero" aria-label="AI/ML Based Cell Tracking System">
        <img
          className="landing-hero-image"
          src="/landing.jpeg"
          alt="Fluorescence microscopy visualization"
        />

        <div className="landing-hero-vignette" />

        <div className="landing-hero-content">
          <div className="landing-hero-kicker">
            LIVE CELL MICROSCOPY / COMPUTATIONAL VISION
          </div>

          <h1>
            <span>AI/ML BASED</span>
            <span>CELL TRACKING SYSTEM</span>
          </h1>

          <p>
            AI-assisted segmentation, tracking, morphology and motility
            analysis for microscopy data.
          </p>

          <button
            type="button"
            className="landing-hero-enter"
            onClick={() =>
              document
                .getElementById("cell-analysis-workspace")
                ?.scrollIntoView({
                  behavior: "smooth",
                  block: "start",
                })
            }
          >
            ENTER ANALYSIS WORKSPACE
            <span aria-hidden="true">↓</span>
          </button>
        </div>

        <div className="landing-hero-corner">
          CELL LAB · LOCAL AI
        </div>
      </section>

      <div id="cell-analysis-workspace">
        <div className="app-shell">

      {/* ======================================================
          SIDEBAR
          ====================================================== */}

      <aside
        className={`sidebar ${
          mobileMenu
            ? "sidebar-open"
            : ""
        }`}
      >
        <div className="brand">

          <div className="brand-mark">
            ✦
          </div>

          <div>
            <div className="brand-name">
              CELL LAB
            </div>

            <div className="brand-subtitle">
              AI MICROSCOPY ANALYSIS
            </div>
          </div>

        </div>


        <nav className="sidebar-nav">

          <button
            onClick={() =>
              scrollToSection(
                "upload"
              )
            }
            className="nav-item"
          >
            <span>
              <Icon
                name="upload"
                size={17}
              />
            </span>

            Upload Data
          </button>


          <button
            onClick={() =>
              scrollToSection(
                "settings"
              )
            }
            className="nav-item"
          >
            <span>
              <Icon
                name="settings"
                size={17}
              />
            </span>

            Analysis Settings
          </button>


          <button
            onClick={() =>
              scrollToSection(
                "results"
              )
            }
            className="nav-item"
          >
            <span>
              <Icon
                name="chart"
                size={17}
              />
            </span>

            Results
          </button>


          <button
            onClick={() =>
              scrollToSection(
                "tracking"
              )
            }
            className="nav-item"
          >
            <span>
              <Icon
                name="tracking"
                size={17}
              />
            </span>

            Cell Tracking
          </button>


          <button
            onClick={() =>
              scrollToSection(
                "advanced"
              )
            }
            className="nav-item"
          >
            <span>
              <Icon
                name="layers"
                size={17}
              />
            </span>

            Scientific Analysis
          </button>


        </nav>


        <div className="sidebar-bottom">

          <div className="local-ai-card">

            <div className="local-ai-status">

              <span className="status-dot" />

              {backendOnline
                ? "LOCAL AI ONLINE"
                : "LOCAL AI OFFLINE"}

            </div>

            <div className="local-ai-text">

              {analysisMode ===
              "microglia"
                ? "Microglia analysis pipeline"
                : "General Cell Lab pipeline"}

            </div>

          </div>


          <div className="model-info">

            <span>
              MODEL
            </span>

            <strong>
              {analysisMode ===
              "microglia"
                ? "Microglia U-Net"
                : "Instance U-Net"}
            </strong>

          </div>

        </div>

      </aside>


      {/* ======================================================
          MAIN
          ====================================================== */}

      <main className="main-content">


        {/* MOBILE HEADER */}

        <div className="mobile-topbar">

          <button
            onClick={() =>
              setMobileMenu(
                !mobileMenu
              )
            }
          >
            ☰
          </button>

          <strong>
            CELL LAB
          </strong>

        </div>


        {/* PAGE HEADER */}

        <header className="page-header">

          <div>

            <div className="eyebrow">
              MICROSCOPY / ANALYSIS WORKSPACE
            </div>

            <h1>
              Cell Analysis Lab
            </h1>

          </div>


          <div className="header-status">

            <span
              className="status-dot"
            />

            {backendOnline
              ? "Local AI"
              : "Backend Offline"}

            <span className="version">

              {jobStatus ===
              "processing"
                ? "PROCESSING"
                : jobStatus ===
                  "completed"
                ? "READY"
                : jobStatus ===
                  "error"
                ? "ERROR"
                : analysisMode ===
                  "microglia"
                ? "MICROGLIA"
                : "v1.0"}

            </span>

          </div>

        </header>


        {/* ====================================================
            MODE SELECTOR
            ==================================================== */}

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(2,minmax(0,1fr))",
            gap: 12,
            marginBottom: 26,
          }}
        >

          <button
            type="button"
            onClick={() =>
              setAnalysisMode(
                "microglia"
              )
            }
            style={{
              textAlign:
                "left",
              padding: 18,
              borderRadius: 12,
              border:
                analysisMode ===
                "microglia"
                  ? "1px solid rgba(83,214,194,.55)"
                  : "1px solid rgba(255,255,255,.08)",
              background:
                analysisMode ===
                "microglia"
                  ? "linear-gradient(145deg, rgba(23,55,58,.92), rgba(16,30,38,.95))"
                  : "rgba(255,255,255,.025)",
              color:
                "inherit",
              cursor:
                "pointer",
            }}
          >

            <div
              style={{
                display:
                  "flex",
                alignItems:
                  "center",
                gap: 9,
                marginBottom:
                  7,
              }}
            >

              <Icon
                name="microscope"
                size={18}
              />

              <strong>
                MICROGLIA MODE
              </strong>

            </div>

            <div
              style={{
                fontSize: 12,
                opacity: .68,
                lineHeight:
                  1.55,
              }}
            >
              Segmentation +
              watershed +
              soma/process
              morphology +
              Sholl +
              motility
            </div>

          </button>


          <button
            type="button"
            onClick={() =>
              setAnalysisMode(
                "bacteria"
              )
            }
            style={{
              textAlign:
                "left",
              padding: 18,
              borderRadius: 12,
              border:
                analysisMode ===
                "bacteria"
                  ? "1px solid rgba(100,181,255,.55)"
                  : "1px solid rgba(255,255,255,.08)",
              background:
                analysisMode ===
                "bacteria"
                  ? "linear-gradient(145deg, rgba(23,42,67,.92), rgba(16,27,38,.95))"
                  : "rgba(255,255,255,.025)",
              color:
                "inherit",
              cursor:
                "pointer",
            }}
          >

            <div
              style={{
                display:
                  "flex",
                alignItems:
                  "center",
                gap: 9,
                marginBottom:
                  7,
              }}
            >

              <Icon
                name="layers"
                size={18}
              />

              <strong>
                BACTERIA / GENERAL CELL MODE
              </strong>

            </div>

            <div
              style={{
                fontSize: 12,
                opacity: .68,
                lineHeight:
                  1.55,
              }}
            >
              Existing Cell Lab
              Instance-U-Net
              segmentation and
              tracking pipeline
            </div>

          </button>

        </div>


        {/* ====================================================
            OVERVIEW
            ==================================================== */}

        <div className="overview-strip">

          <div className="overview-item">

            <span className="overview-icon">
              <Icon
                name="chart"
                size={17}
              />
            </span>

            <div>

              <span>
                {analysisMode ===
                "microglia"
                  ? "CELLS TRACKED"
                  : "TRACKED CELLS"}
              </span>

              <strong>

                {formatInteger(
                  analysisMode ===
                  "microglia"
                    ? microgliaStats.cellsTracked
                    : trackCount
                )}

              </strong>

            </div>

          </div>


          <div className="overview-divider" />


          <div className="overview-item">

            <span className="overview-icon">
              <Icon
                name={
                  analysisMode ===
                  "microglia"
                    ? "pulse"
                    : "tracking"
                }
                size={17}
              />
            </span>

            <div>

              <span>
                {analysisMode ===
                "microglia"
                  ? "PROCESS-TIP SPEED"
                  : "MEAN SPEED"}
              </span>

              <strong>

                {analysisMode ===
                "microglia"
                  ? microgliaStats.meanTipSpeed !==
                    null
                    ? formatNumber(
                        microgliaStats.meanTipSpeed
                      )
                    : "—"
                  : formatNumber(
                      bacteriaStatistics.meanSpeed
                    )}

                <small>
                  {" "}
                  {analysisMode ===
                  "microglia"
                    ? "µm/min"
                    : "px/s"}
                </small>

              </strong>

            </div>

          </div>


          <div className="overview-divider" />


          <div className="overview-item">

            <span className="overview-icon">
              <Icon
                name="layers"
                size={17}
              />
            </span>

            <div>

              <span>
                ANALYSIS STATUS
              </span>

              <strong
                className="overview-ready"
              >
                {isCompleted
                  ? "READY"
                  : jobStatus ===
                    "processing"
                  ? "RUNNING"
                  : jobStatus ===
                    "error"
                  ? "ERROR"
                  : "WAITING"}
              </strong>

            </div>

          </div>

        </div>


        {/* ====================================================
            UPLOAD
            ==================================================== */}

        <section
          id="upload"
          className="section"
        >

          <SectionHeading
            number="01"
            title="Upload Microscopy Data"
            description={
              analysisMode ===
              "microglia"
                ? "Upload a microglia time-lapse, TIFF stack, fluorescence image sequence or microscopy video. The local backend performs segmentation, instance separation, morphology analysis and motility estimation."
                : "Upload your microscopy video and run the existing Cell Lab quantitative analysis pipeline."
            }
          />


          <label
            className="upload-zone"
            onDragOver={(
              event
            ) =>
              event.preventDefault()
            }
            onDrop={
              handleDrop
            }
          >

            <input
              type="file"
              accept=".avi,.mp4,.mov,.mkv,.webm,.mpg,.mpeg,.tif,.tiff,.png,.jpg,.jpeg,video/*,image/tiff,image/png,image/jpeg"
              onChange={
                handleFiles
              }
            />

            <div className="upload-icon">
              <Icon
                name="upload"
                size={26}
              />
            </div>

            <h3>
              Drop your microscopy data here
            </h3>

            <p>
              or click to browse your computer
            </p>

            <div className="format-list">

              <span>
                SUPPORTED
              </span>

              <b>AVI</b>
              <b>MP4</b>
              <b>MOV</b>
              <b>MKV</b>
              <b>TIFF</b>

            </div>

          </label>


          {files.length > 0 && (
            <div className="uploaded-files">

              <div className="file-list-header">

                <span>
                  Selected file

                  <span className="count-badge">
                    1
                  </span>
                </span>

                <button
                  className="clear-button"
                  onClick={
                    clearFiles
                  }
                >
                  Clear
                </button>

              </div>


              <div className="file-row">

                <div className="file-type-icon">
                  <Icon
                    name={
                      files[0].name
                        .toLowerCase()
                        .endsWith(
                          ".tif"
                        ) ||
                      files[0].name
                        .toLowerCase()
                        .endsWith(
                          ".tiff"
                        )
                        ? "layers"
                        : "play"
                    }
                    size={17}
                  />
                </div>


                <div className="file-info">

                  <strong>
                    {files[0].name}
                  </strong>

                  <span>
                    {formatBytes(
                      files[0].size
                    )}

                    {" · "}

                    {files[0].name
                      .toLowerCase()
                      .match(
                        /\.(tif|tiff)$/i
                      )
                      ? "TIFF stack / microscopy image"
                      : "ready for local analysis"}
                  </span>

                </div>


                <span className="file-ready">
                  READY
                </span>


                <button
                  className="remove-file"
                  onClick={
                    clearFiles
                  }
                >
                  ×
                </button>

              </div>

            </div>
          )}


          <div className="analysis-control">

            <div>

              <span className="analysis-control-label">
                {jobStatus ===
                "processing"
                  ? "ANALYSIS RUNNING"
                  : "ANALYSIS CONTROL"}
              </span>

              <strong>
                {progressMessage ||
                  (
                    files.length
                      ? `${analysisMode === "microglia" ? "Microglia" : "General Cell"} mode selected — start the AI analysis`
                      : "Select a microscopy file to begin"
                  )}
              </strong>

            </div>


            <button
              className="primary-button"
              disabled={
                !files.length ||
                jobStatus ===
                  "processing" ||
                jobStatus ===
                  "uploading"
              }
              onClick={
                analyzeVideo
              }
            >

              {jobStatus ===
              "processing"
                ? "ANALYZING…"
                : jobStatus ===
                  "uploading"
                ? "UPLOADING…"
                : "START AI ANALYSIS"}

              <span>
                →
              </span>

            </button>

          </div>


          {dataError && (
            <div
              className="data-warning"
              style={{
                marginTop: 15,
              }}
            >
              {dataError}
            </div>
          )}

        </section>


        {/* ====================================================
            SETTINGS
            ==================================================== */}

        <section
          id="settings"
          className="section"
        >

          <SectionHeading
            number="02"
            title="Analysis Settings"
            description={
              analysisMode ===
              "microglia"
                ? "Microglia-specific controls for calibration, temporal sampling and inference speed."
                : "Controls for the existing Cell Lab analysis pipeline."
            }
          />


          <div className="settings-grid">


            <div className="setting-card">

              <label>
                Pixel Size
              </label>

              <div className="input-with-unit">

                <input
                  type="number"
                  min="0"
                  step="0.001"
                  placeholder="Optional"
                  value={
                    settings.pixelSize
                  }
                  onChange={(
                    event
                  ) =>
                    updateSetting(
                      "pixelSize",
                      event.target.value
                    )
                  }
                />

                <span>
                  µm/px
                </span>

              </div>

              <small>
                {analysisMode ===
                "microglia"
                  ? "Required for physical morphology and motility units."
                  : "Optional spatial calibration."}
              </small>

            </div>


            <div className="setting-card">

              <label>
                Process Every
              </label>

              <div className="input-with-unit">

                <input
                  type="number"
                  min="1"
                  max="10"
                  step="1"
                  value={
                    settings.processEveryNFrames
                  }
                  onChange={(
                    event
                  ) =>
                    updateSetting(
                      "processEveryNFrames",
                      event.target.value
                    )
                  }
                />

                <span>
                  frames
                </span>

              </div>

              <small>
                1 = maximum temporal detail; higher values trade temporal resolution for speed.
              </small>

            </div>


            <div className="setting-card">

              <label>
                Inference Batch
              </label>

              <div className="input-with-unit">

                <input
                  type="number"
                  min="1"
                  max="8"
                  step="1"
                  value={
                    settings.inferenceBatchSize
                  }
                  onChange={(
                    event
                  ) =>
                    updateSetting(
                      "inferenceBatchSize",
                      event.target.value
                    )
                  }
                />

                <span>
                  frames
                </span>

              </div>

              <small>
                Batch size used by the local AI inference pipeline.
              </small>

            </div>


            <div className="setting-card">

              <label>
                Frame Interval
              </label>

              <div className="input-with-unit">

                <input
                  type="number"
                  min="0"
                  step="0.001"
                  placeholder="Unknown"
                  value={
                    settings.frameInterval
                  }
                  onChange={(
                    event
                  ) =>
                    updateSetting(
                      "frameInterval",
                      event.target.value
                    )
                  }
                />

                <span>
                  sec
                </span>

              </div>

              <small>
                {analysisMode ===
                "microglia"
                  ? "Enter the actual acquisition interval to obtain meaningful µm/min velocities."
                  : "Optional temporal calibration."}
              </small>

            </div>


          </div>


          {analysisMode ===
            "microglia" && (
            <div
              style={{
                marginTop: 16,
                padding: 15,
                border:
                  "1px solid rgba(83,214,194,.15)",
                borderRadius: 10,
                background:
                  "rgba(83,214,194,.035)",
                fontSize: 12,
                lineHeight: 1.6,
                opacity: 0.76,
              }}
            >
              <strong>
                Microglia pipeline:
              </strong>{" "}
              U-Net semantic segmentation →
              distance-transform / watershed
              instance separation → soma proxy →
              skeleton topology → Sholl →
              process-tip matching → extension /
              retraction → motility metrics.
            </div>
          )}

        </section>


        {/* ====================================================
            RESULTS
            ==================================================== */}

        <section
          id="results"
          className="section"
        >

          <SectionHeading
            number="03"
            title="Quantitative Results"
            description={
              analysisMode ===
              "microglia"
                ? "Microglia morphology, arborization, tracking and motility metrics from the current analysis job."
                : "Quantitative results loaded dynamically from the current analysis job."
            }
          />


          {jobStatus ===
            "processing" && (
            <ProgressPanel
              progress={
                analysisProgress
              }
              message={
                progressMessage
              }
            />
          )}


          {jobStatus !==
            "processing" && (
            <>

              {analysisMode ===
              "microglia" ? (

                <>

                  <div className="stats-grid">

                    <StatCard
                      label="CELLS TRACKED"
                      value={formatInteger(
                        microgliaStats.cellsTracked
                      )}
                      detail={`${formatInteger(
                        microgliaStats.cellObservations
                      )} cell/frame observations`}
                      accent
                    />

                    <StatCard
                      label="MEAN CELL AREA"
                      value={formatNumber(
                        microgliaStats.meanArea
                      )}
                      unit="µm²"
                      detail="Projected segmented area"
                    />

                    <StatCard
                      label="PROCESS LENGTH"
                      value={formatNumber(
                        microgliaStats.meanProcessLength
                      )}
                      unit="µm"
                      detail="Mean skeleton/arbor length"
                    />

                    <StatCard
                      label="BRANCH POINTS"
                      value={formatNumber(
                        microgliaStats.meanBranchPoints
                      )}
                      unit="junctions"
                      detail="Per cell observation"
                    />

                    <StatCard
                      label="ENDPOINTS"
                      value={formatNumber(
                        microgliaStats.meanEndpoints
                      )}
                      unit="ends"
                      detail="Detected process endpoints"
                    />

                    <StatCard
                      label="SHOLL MAX"
                      value={formatNumber(
                        microgliaStats.meanSholl
                      )}
                      unit="intersections"
                      detail="Mean maximum Sholl intersections"
                    />

                    <StatCard
                      label="SOMA SPEED"
                      value={
                        microgliaStats.meanSomaSpeed !==
                        null
                          ? formatNumber(
                              microgliaStats.meanSomaSpeed
                            )
                          : "—"
                      }
                      unit="µm/min"
                      detail={
                        settings.frameInterval
                          ? "Using supplied frame interval"
                          : "Temporal interval not supplied"
                      }
                    />

                    <StatCard
                      label="PROCESS-TIP SPEED"
                      value={
                        microgliaStats.meanTipSpeed !==
                        null
                          ? formatNumber(
                              microgliaStats.meanTipSpeed
                            )
                          : "—"
                      }
                      unit="µm/min"
                      detail="Matched endpoint displacement"
                    />

                  </div>


                  <div
                    className="experiment-card"
                  >

                    <div className="experiment-header">

                      <div>

                        <span>
                          MICROGLIA EXPERIMENT
                        </span>

                        <h3>
                          {videoName}
                        </h3>

                      </div>

                      <div className="experiment-badge">
                        {isCompleted
                          ? "ANALYZED"
                          : "NOT ANALYZED"}
                      </div>

                    </div>


                    <div className="experiment-grid">

                      <InfoItem
                        label="Resolution"
                        value={
                          width !==
                            "—" &&
                          height !==
                            "—"
                            ? `${width} × ${height} px`
                            : "Unavailable"
                        }
                      />

                      <InfoItem
                        label="Frames"
                        value={formatInteger(
                          frameCount
                        )}
                      />

                      <InfoItem
                        label="Source FPS"
                        value={`${formatNumber(
                          actualFps,
                          3
                        )} FPS`}
                      />

                      <InfoItem
                        label="Duration"
                        value={`${formatNumber(
                          durationSeconds,
                          2
                        )} s`}
                      />

                      <InfoItem
                        label="Pixel calibration"
                        value={
                          pixelSize
                            ? `${formatNumber(
                                pixelSize,
                                3
                              )} µm/px`
                            : "Not provided"
                        }
                      />

                      <InfoItem
                        label="Frame step"
                        value={`${processStep} source frame(s)`}
                      />

                      <InfoItem
                        label="Batch"
                        value={`${inferenceBatch} frame(s)`}
                      />

                      <InfoItem
                        label="Model"
                        value={
                          microgliaInfo?.model
                            ? "Microglia checkpoint"
                            : "Microglia U-Net"
                        }
                      />

                    </div>

                  </div>


                  <MicrogliaInterpretation
                    microglia={{
                      mean_cell_area_um2:
                        microgliaStats.meanArea,
                      mean_soma_area_proxy_um2:
                        microgliaStats.meanSomaArea,
                      mean_skeleton_length_um:
                        microgliaStats.meanProcessLength,
                      mean_branch_points:
                        microgliaStats.meanBranchPoints,
                      mean_endpoints:
                        microgliaStats.meanEndpoints,
                      mean_sholl_max_intersections:
                        microgliaStats.meanSholl,
                      mean_soma_speed_um_min:
                        microgliaStats.meanSomaSpeed,
                      mean_process_tip_speed_um_min:
                        microgliaStats.meanTipSpeed,
                      process_extension_events:
                        microgliaStats.extensions,
                      process_retraction_events:
                        microgliaStats.retractions,
                    }}
                    population={{
                      minimum_cell_count:
                        populationMinimum,
                      maximum_cell_count:
                        populationMaximum,
                    }}
                    experiment={
                      experiment
                    }
                  />

                </>

              ) : (

                <div className="stats-grid">

                  <StatCard
                    label="AVERAGE CELLS / FRAME"
                    value={formatNumber(
                      bacteriaStatistics.averageCells
                    )}
                    detail={`Peak detected cells: ${formatInteger(
                      bacteriaStatistics.maximumCells
                    )}`}
                    accent
                  />

                  <StatCard
                    label="AVERAGE MOVING"
                    value={formatNumber(
                      bacteriaStatistics.averageMoving
                    )}
                    detail="Moving cells per frame"
                  />

                  <StatCard
                    label="MEAN SPEED"
                    value={formatNumber(
                      bacteriaStatistics.meanSpeed
                    )}
                    unit="px/s"
                    detail={`Median: ${formatNumber(
                      bacteriaStatistics.medianSpeed
                    )} px/s`}
                  />

                  <StatCard
                    label="MAXIMUM SPEED"
                    value={formatNumber(
                      bacteriaStatistics.maximumSpeed
                    )}
                    unit="px/s"
                    detail="Maximum observed frame speed"
                  />

                  <StatCard
                    label="MEAN DISPLACEMENT"
                    value={formatNumber(
                      bacteriaStatistics.meanDisplacement
                    )}
                    unit="px"
                    detail="Frame-to-frame movement"
                  />

                  <StatCard
                    label="MEAN CELL AREA"
                    value={formatNumber(
                      bacteriaStatistics.meanArea
                    )}
                    unit="px²"
                    detail="Average segmented area"
                  />

                  <StatCard
                    label="TRACKED OBJECTS"
                    value={formatInteger(
                      trackCount
                    )}
                    detail="Individual cell tracks"
                  />

                  <StatCard
                    label="TOTAL DISPLACEMENT"
                    value={formatNumber(
                      bacteriaStatistics.totalDisplacement
                    )}
                    unit="px"
                    detail="Sum of measured movement"
                  />

                </div>

              )}

            </>
          )}

        </section>


        {/* ====================================================
            ANALYZED VIDEO
            ==================================================== */}

        <section
          id="video"
          className="section"
        >

          <SectionHeading
            number="04"
            title="Analyzed Output"
            description="The annotated output generated by the Python backend for the current analysis job."
          />


          <div className="video-result-card">

            <div
              className="video-placeholder"
            >

              {annotatedVideoUrl ? (

                <>

                  <video
                    controls
                    style={{
                      width:
                        "100%",
                      maxHeight:
                        460,
                      borderRadius:
                        12,
                      background:
                        "#000",
                    }}
                    src={
                      annotatedVideoUrl
                    }
                  />

                  <div className="video-file-name">
                    {annotatedVideoFile}
                  </div>

                  <a
                    className="secondary-button"
                    href={
                      annotatedVideoUrl
                    }
                    download
                  >
                    Open / Download Annotated Output
                  </a>

                </>

              ) : (

                <>

                  <div className="video-placeholder-icon">
                    <Icon
                      name="play"
                      size={28}
                    />
                  </div>

                  <h3>
                    Annotated output
                  </h3>

                  <p>
                    Run an analysis to generate the processed output.
                  </p>

                </>

              )}

            </div>


            <div
              className="video-data-panel"
            >

              {analysisMode ===
              "microglia" ? (

                <>

                  <div className="video-data-title">
                    MICROGLIA METRICS
                  </div>

                  <VideoMetricMicroglia
                    label="Cells tracked"
                    value={formatInteger(
                      microgliaStats.cellsTracked
                    )}
                    suffix="tracks"
                  />

                  <VideoMetricMicroglia
                    label="Mean arbor length"
                    value={formatNumber(
                      microgliaStats.meanProcessLength
                    )}
                    suffix="µm"
                  />

                  <VideoMetricMicroglia
                    label="Branch points"
                    value={formatNumber(
                      microgliaStats.meanBranchPoints
                    )}
                    suffix="per cell"
                  />

                  <VideoMetricMicroglia
                    label="Extensions"
                    value={formatInteger(
                      microgliaStats.extensions
                    )}
                    suffix="events"
                  />

                  <VideoMetricMicroglia
                    label="Retractions"
                    value={formatInteger(
                      microgliaStats.retractions
                    )}
                    suffix="events"
                  />

                </>

              ) : (

                <>

                  <div className="video-data-title">
                    FRAME-LEVEL METRICS
                  </div>

                  <VideoMetricMicroglia
                    label="Cells detected"
                    value={formatNumber(
                      bacteriaStatistics.averageCells
                    )}
                    suffix="avg/frame"
                  />

                  <VideoMetricMicroglia
                    label="Moving cells"
                    value={formatNumber(
                      bacteriaStatistics.averageMoving
                    )}
                    suffix="avg/frame"
                  />

                  <VideoMetricMicroglia
                    label="Mean speed"
                    value={formatNumber(
                      bacteriaStatistics.meanSpeed
                    )}
                    suffix="px/s"
                  />

                  <VideoMetricMicroglia
                    label="Maximum speed"
                    value={formatNumber(
                      bacteriaStatistics.maximumSpeed
                    )}
                    suffix="px/s"
                  />

                </>

              )}

            </div>

          </div>

        </section>


        {/* ====================================================
            ADVANCED / GRAPHS
            ==================================================== */}

        <section
          id="advanced"
          className="section"
        >

          <SectionHeading
            number="05"
            title={
              analysisMode ===
              "microglia"
                ? "Scientific Analysis"
                : "Advanced Motion Analysis"
            }
            description={
              analysisMode ===
              "microglia"
                ? "Graphs are generated from the same Microglia analysis job as the quantitative tables."
                : "Graphs are generated from the same Cell Lab analysis job as the CSV outputs."
            }
          />


          {graphCategories.length >
            1 && (
            <div className="graph-filter">

              {graphCategories.map(
                (
                  category
                ) => (
                  <button
                    key={
                      category
                    }
                    className={
                      activeGraphCategory ===
                      category
                        ? "graph-filter-active"
                        : ""
                    }
                    onClick={() =>
                      setActiveGraphCategory(
                        category
                      )
                    }
                  >
                    {category}
                  </button>
                )
              )}

            </div>
          )}


          <div className="analysis-graph-grid">

            {isCompleted &&
            filteredGraphs.length ? (

              filteredGraphs.map(
                (
                  graph
                ) => (
                  <GraphCard
                    key={
                      graph.key
                    }
                    title={
                      graph.title
                    }
                    description={
                      graph.description
                    }
                    file={
                      graph.file
                    }
                    baseUrl={
                      resultBase
                    }
                  />
                )
              )

            ) : (

              <div
                className="loading-card"
                style={{
                  gridColumn:
                    "1 / -1",
                }}
              >
                {jobStatus ===
                "processing"
                  ? "Graphs will appear when analysis finishes."
                  : "Run an analysis to generate the quantitative visualizations."}
              </div>

            )}

          </div>

        </section>


        {/* ====================================================
            POPULATION
            ==================================================== */}

        {analysisMode ===
          "microglia" &&
          isCompleted && (
          <section className="section">

            <SectionHeading
              number="06"
              title="Population & Process Dynamics"
              description="Imaging-derived population and process-event measurements."
            />


            <div className="stats-grid">

              <StatCard
                label="AVERAGE CELL COUNT"
                value={formatNumber(
                  populationAverage
                )}
                unit="cells/frame"
              />

              <StatCard
                label="MINIMUM COUNT"
                value={formatInteger(
                  populationMinimum
                )}
                unit="cells"
              />

              <StatCard
                label="MAXIMUM COUNT"
                value={formatInteger(
                  populationMaximum
                )}
                unit="cells"
              />

              <StatCard
                label="EXTENSION EVENTS"
                value={formatInteger(
                  microgliaStats.extensions
                )}
                unit="events"
              />

              <StatCard
                label="RETRACTION EVENTS"
                value={formatInteger(
                  microgliaStats.retractions
                )}
                unit="events"
              />

              <StatCard
                label="DISAPPEARANCE-LIKE"
                value={formatInteger(
                  microgliaStats.disappearanceCandidates
                )}
                unit="candidates"
                detail="Candidate imaging event, not apoptosis diagnosis"
              />

              <StatCard
                label="APPARENT GROWTH"
                value={
                  populationSummary.apparent_growth_rate_per_min !==
                  null &&
                  populationSummary.apparent_growth_rate_per_min !==
                  undefined
                    ? formatNumber(
                        populationSummary.apparent_growth_rate_per_min,
                        4
                      )
                    : "—"
                }
                unit="/min"
                detail="Imaging-derived population estimate"
              />

              <StatCard
                label="DOUBLING TIME"
                value={
                  populationSummary.estimated_doubling_time_min !==
                  null &&
                  populationSummary.estimated_doubling_time_min !==
                  undefined
                    ? formatNumber(
                        populationSummary.estimated_doubling_time_min,
                        2
                      )
                    : "—"
                }
                unit="min"
                detail="Only meaningful for suitable long time-lapse data"
              />

            </div>


            {populationRows.length >
              0 && (
              <div
                className="experiment-card"
                style={{
                  marginTop: 18,
                }}
              >

                <div className="experiment-header">

                  <div>
                    <span>
                      POPULATION TIME SERIES
                    </span>

                    <h3>
                      Cell count over time
                    </h3>
                  </div>

                </div>


                <div
                  style={{
                    overflowX:
                      "auto",
                  }}
                >

                  <table className="tracking-table">

                    <thead>
                      <tr>
                        <th>
                          Time
                        </th>

                        <th>
                          Cell Count
                        </th>
                      </tr>
                    </thead>

                    <tbody>

                      {populationRows
                        .slice(
                          0,
                          100
                        )
                        .map(
                          (
                            row,
                            index
                          ) => (
                            <tr
                              key={`${row.time_seconds}-${index}`}
                            >

                              <td>
                                {formatNumber(
                                  row.time_seconds,
                                  2
                                )}{" "}
                                s
                              </td>

                              <td>
                                {formatInteger(
                                  row.cell_count
                                )}
                              </td>

                            </tr>
                          )
                        )}

                    </tbody>

                  </table>

                </div>

              </div>
            )}

          </section>
        )}


        {/* ====================================================
            CELL PERFORMANCE
            ==================================================== */}

        <section className="section">

          <SectionHeading
            number={
              analysisMode ===
              "microglia"
                ? "07"
                : "06"
            }
            title="Cell Performance"
            description={
              analysisMode ===
              "microglia"
                ? "Ranked microglia tracks from the current analysis."
                : "Ranked cells from the current analysis."
            }
          />


          <div className="ranking-grid">

            {analysisMode ===
            "microglia" ? (

              <>

                <RankingPanelDynamic
                  title="Fastest Soma"
                  subtitle="Highest mean soma speed"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.mean_soma_speed_um_min
                        ) -
                        safeNumber(
                          a.mean_soma_speed_um_min
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="mean_soma_speed_um_min"
                  unit="µm/min"
                  onSelect={
                    setSelectedCell
                  }
                />


                <RankingPanelDynamic
                  title="Fastest Process Tips"
                  subtitle="Highest mean process-tip speed"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.mean_process_tip_speed_um_min
                        ) -
                        safeNumber(
                          a.mean_process_tip_speed_um_min
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="mean_process_tip_speed_um_min"
                  unit="µm/min"
                  onSelect={
                    setSelectedCell
                  }
                />


                <RankingPanelDynamic
                  title="Longest Arbor"
                  subtitle="Largest mean skeleton length"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.mean_skeleton_length_um
                        ) -
                        safeNumber(
                          a.mean_skeleton_length_um
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="mean_skeleton_length_um"
                  unit="µm"
                  onSelect={
                    setSelectedCell
                  }
                />


                <RankingPanelDynamic
                  title="Most Branching"
                  subtitle="Highest mean branch-point count"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.mean_branch_points
                        ) -
                        safeNumber(
                          a.mean_branch_points
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="mean_branch_points"
                  unit="junctions"
                  onSelect={
                    setSelectedCell
                  }
                />

              </>

            ) : (

              <>

                <RankingPanelDynamic
                  title="Fastest Average"
                  subtitle="Highest average speed"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.average_speed_px_s
                        ) -
                        safeNumber(
                          a.average_speed_px_s
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="average_speed_px_s"
                  unit="px/s"
                  onSelect={
                    setSelectedCell
                  }
                />


                <RankingPanelDynamic
                  title="Fastest Peak"
                  subtitle="Highest maximum speed"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.maximum_speed_px_s
                        ) -
                        safeNumber(
                          a.maximum_speed_px_s
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="maximum_speed_px_s"
                  unit="px/s"
                  onSelect={
                    setSelectedCell
                  }
                />


                <RankingPanelDynamic
                  title="Most Distance"
                  subtitle="Largest total distance"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.total_distance_px
                        ) -
                        safeNumber(
                          a.total_distance_px
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="total_distance_px"
                  unit="px"
                  onSelect={
                    setSelectedCell
                  }
                />


                <RankingPanelDynamic
                  title="Longest Tracks"
                  subtitle="Most frames tracked"
                  data={[
                    ...summaryData,
                  ]
                    .sort(
                      (a, b) =>
                        safeNumber(
                          b.frames_tracked
                        ) -
                        safeNumber(
                          a.frames_tracked
                        )
                    )
                    .slice(
                      0,
                      5
                    )}
                  field="frames_tracked"
                  unit="frames"
                  onSelect={
                    setSelectedCell
                  }
                />

              </>

            )}

          </div>

        </section>


        {/* ====================================================
            TRACKING
            ==================================================== */}

        <section
          id="tracking"
          className="section"
        >

          <SectionHeading
            number={
              analysisMode ===
              "microglia"
                ? "08"
                : "07"
            }
            title={
              analysisMode ===
              "microglia"
                ? "Microglia Tracking"
                : "Cell Tracking"
            }
            description={
              analysisMode ===
              "microglia"
                ? "Individual microglia tracks with soma motion, process morphology and process-tip dynamics."
                : "Individual cell movement and morphology extracted from the current analysis."
            }
          />


          <div className="tracking-summary">

            <div>

              <span>
                INDIVIDUAL CELL DATA
              </span>

              <h3>
                {formatInteger(
                  summaryData.length
                )}{" "}
                tracked objects
              </h3>

            </div>


            <div className="tracking-controls">

              <div className="tracking-search">

                <Icon
                  name="search"
                  size={17}
                />

                <input
                  placeholder="Search cell ID…"
                  value={
                    search
                  }
                  onChange={(
                    event
                  ) =>
                    setSearch(
                      event.target.value
                    )
                  }
                />

              </div>


              <select
                value={
                  sortBy
                }
                onChange={(
                  event
                ) =>
                  setSortBy(
                    event.target.value
                  )
                }
              >

                {trackingSortOptions.map(
                  (
                    option
                  ) => (
                    <option
                      key={
                        option.value
                      }
                      value={
                        option.value
                      }
                    >
                      {option.label}
                    </option>
                  )
                )}

              </select>

            </div>

          </div>


          <div className="tracking-table-wrapper">

            {analysisMode ===
            "microglia" ? (

              <table className="tracking-table">

                <thead>

                  <tr>

                    <th>
                      Track
                    </th>

                    <th>
                      Frames
                    </th>

                    <th>
                      Duration
                    </th>

                    <th>
                      Soma Speed
                    </th>

                    <th>
                      Tip Speed
                    </th>

                    <th>
                      Arbor
                    </th>

                    <th>
                      Branches
                    </th>

                    <th>
                      Extensions
                    </th>

                    <th>
                      Retractions
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {filteredSummary
                    .slice(
                      0,
                      100
                    )
                    .map(
                      (
                        cell,
                        index
                      ) => {

                        const trackId =
                          cell.track_id ??
                          cell.cell_id;

                        return (
                          <tr
                            key={`${trackId}-${index}`}
                            className={
                              selectedCell &&
                              String(
                                selectedCell.track_id ??
                                  selectedCell.cell_id
                              ) ===
                                String(
                                  trackId
                                )
                                ? "selected-row"
                                : ""
                            }
                            onClick={() =>
                              setSelectedCell(
                                cell
                              )
                            }
                          >

                            <td>

                              <div className="cell-name">

                                <span className="cell-dot" />

                                #
                                {
                                  trackId
                                }

                              </div>

                            </td>


                            <td>
                              {formatInteger(
                                cell.frames_tracked
                              )}
                            </td>


                            <td>
                              {formatNumber(
                                cell.duration_seconds
                              )}{" "}
                              s
                            </td>


                            <td className="speed-highlight">

                              {hasNumber(
                                cell.mean_soma_speed_um_min
                              )
                                ? formatNumber(
                                    cell.mean_soma_speed_um_min
                                  )
                                : "—"}

                              {" "}
                              µm/min

                            </td>


                            <td>

                              {hasNumber(
                                cell.mean_process_tip_speed_um_min
                              )
                                ? formatNumber(
                                    cell.mean_process_tip_speed_um_min
                                  )
                                : "—"}

                              {" "}
                              µm/min

                            </td>


                            <td>

                              {formatNumber(
                                cell.mean_skeleton_length_um
                              )}{" "}
                              µm

                            </td>


                            <td>
                              {formatNumber(
                                cell.mean_branch_points
                              )}
                            </td>


                            <td>
                              {formatInteger(
                                cell.extension_events
                              )}
                            </td>


                            <td>
                              {formatInteger(
                                cell.retraction_events
                              )}
                            </td>

                          </tr>
                        );
                      }
                    )}

                </tbody>

              </table>

            ) : (

              <table className="tracking-table">

                <thead>

                  <tr>

                    <th>
                      Cell
                    </th>

                    <th>
                      Frames
                    </th>

                    <th>
                      Duration
                    </th>

                    <th>
                      Avg Speed
                    </th>

                    <th>
                      Max Speed
                    </th>

                    <th>
                      Total Distance
                    </th>

                    <th>
                      Net Displacement
                    </th>

                    <th>
                      Avg Area
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {filteredSummary
                    .slice(
                      0,
                      100
                    )
                    .map(
                      (
                        cell,
                        index
                      ) => {

                        const trackId =
                          cell.track_id ??
                          cell.cell_id;

                        return (
                          <tr
                            key={`${trackId}-${index}`}
                            className={
                              selectedCell &&
                              String(
                                selectedCell.track_id ??
                                  selectedCell.cell_id
                              ) ===
                                String(
                                  trackId
                                )
                                ? "selected-row"
                                : ""
                            }
                            onClick={() =>
                              setSelectedCell(
                                cell
                              )
                            }
                          >

                            <td>

                              <div className="cell-name">

                                <span className="cell-dot" />

                                #
                                {
                                  trackId
                                }

                              </div>

                            </td>


                            <td>
                              {formatInteger(
                                cell.frames_tracked
                              )}
                            </td>


                            <td>
                              {formatNumber(
                                cell.duration_seconds
                              )}{" "}
                              s
                            </td>


                            <td className="speed-highlight">

                              {formatNumber(
                                cell.average_speed_px_s
                              )}{" "}
                              px/s

                            </td>


                            <td>

                              {formatNumber(
                                cell.maximum_speed_px_s
                              )}{" "}
                              px/s

                            </td>


                            <td>

                              {formatNumber(
                                cell.total_distance_px
                              )}{" "}
                              px

                            </td>


                            <td>

                              {formatNumber(
                                cell.net_displacement_px
                              )}{" "}
                              px

                            </td>


                            <td>

                              {formatNumber(
                                cell.average_area_px
                              )}{" "}
                              px²

                            </td>

                          </tr>
                        );
                      }
                    )}

                </tbody>

              </table>

            )}


            {!filteredSummary.length && (
              <div className="empty-table">
                No cell tracking data available yet.
              </div>
            )}

          </div>

        </section>


        {/* ====================================================
            SELECTED CELL
            ==================================================== */}

        {selectedCell && (
          <section className="section">

            <div className="selected-cell-card">

              <div className="selected-cell-header">

                <div>

                  <span>
                    SELECTED{" "}
                    {analysisMode ===
                    "microglia"
                      ? "MICROGLIA"
                      : "CELL"}
                  </span>

                  <h3>

                    Cell #
                    {
                      selectedCell.track_id ??
                      selectedCell.cell_id
                    }

                  </h3>

                </div>


                <button
                  onClick={() =>
                    setSelectedCell(
                      null
                    )
                  }
                >
                  Close
                </button>

              </div>


              {analysisMode ===
              "microglia" ? (

                <>

                  <div className="selected-cell-stats">

                    <StatCard
                      label="SOMA SPEED"
                      value={
                        hasNumber(
                          selectedCell.mean_soma_speed_um_min
                        )
                          ? formatNumber(
                              selectedCell.mean_soma_speed_um_min
                            )
                          : "—"
                      }
                      unit="µm/min"
                    />

                    <StatCard
                      label="TIP SPEED"
                      value={
                        hasNumber(
                          selectedCell.mean_process_tip_speed_um_min
                        )
                          ? formatNumber(
                              selectedCell.mean_process_tip_speed_um_min
                            )
                          : "—"
                      }
                      unit="µm/min"
                    />

                    <StatCard
                      label="ARBOR LENGTH"
                      value={formatNumber(
                        selectedCell.mean_skeleton_length_um
                      )}
                      unit="µm"
                    />

                    <StatCard
                      label="BRANCH POINTS"
                      value={formatNumber(
                        selectedCell.mean_branch_points
                      )}
                      unit="junctions"
                    />

                    <StatCard
                      label="EXTENSIONS"
                      value={formatInteger(
                        selectedCell.extension_events
                      )}
                    />

                    <StatCard
                      label="RETRACTIONS"
                      value={formatInteger(
                        selectedCell.retraction_events
                      )}
                    />

                  </div>


                  <div className="position-panel">

                    <div>

                      <span>
                        START / END
                      </span>

                      <strong>
                        {
                          selectedCell.start_frame ??
                          "—"
                        }
                        {" → "}
                        {
                          selectedCell.end_frame ??
                          "—"
                        }
                      </strong>

                    </div>


                    <div>

                      <span>
                        DURATION
                      </span>

                      <strong>
                        {formatNumber(
                          selectedCell.duration_seconds
                        )}{" "}
                        s
                      </strong>

                    </div>

                  </div>


                  {selectedTrack.length >
                    0 && (
                    <div className="mini-track-table">

                      <div className="mini-track-title">
                        SOMA / CELL MOVEMENT
                      </div>

                      <table>

                        <thead>

                          <tr>

                            <th>
                              Frame
                            </th>

                            <th>
                              Time
                            </th>

                            <th>
                              Soma X
                            </th>

                            <th>
                              Soma Y
                            </th>

                            <th>
                              Area
                            </th>

                            <th>
                              Soma Speed
                            </th>

                          </tr>

                        </thead>


                        <tbody>

                          {selectedTrack
                            .slice(
                              0,
                              60
                            )
                            .map(
                              (
                                row,
                                index
                              ) => (
                                <tr
                                  key={`${row.frame}-${index}`}
                                >

                                  <td>
                                    {
                                      row.frame
                                    }
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.time_seconds
                                    )}{" "}
                                    s
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.soma_x_um
                                    )}{" "}
                                    µm
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.soma_y_um
                                    )}{" "}
                                    µm
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.area_um2
                                    )}{" "}
                                    µm²
                                  </td>

                                  <td>
                                    {hasNumber(
                                      row.soma_speed_um_min
                                    )
                                      ? `${formatNumber(
                                          row.soma_speed_um_min
                                        )} µm/min`
                                      : "—"}
                                  </td>

                                </tr>
                              )
                            )}

                        </tbody>

                      </table>

                    </div>
                  )}


                  {selectedTipRows.length >
                    0 && (
                    <div
                      className="mini-track-table"
                      style={{
                        marginTop: 18,
                      }}
                    >

                      <div className="mini-track-title">
                        PROCESS-TIP EVENTS
                      </div>


                      <table>

                        <thead>

                          <tr>

                            <th>
                              From
                            </th>

                            <th>
                              To
                            </th>

                            <th>
                              Displacement
                            </th>

                            <th>
                              Radial Change
                            </th>

                            <th>
                              State
                            </th>

                            <th>
                              Speed
                            </th>

                          </tr>

                        </thead>


                        <tbody>

                          {selectedTipRows
                            .slice(
                              0,
                              80
                            )
                            .map(
                              (
                                row,
                                index
                              ) => (
                                <tr
                                  key={`${row.frame_from}-${row.frame_to}-${index}`}
                                >

                                  <td>
                                    {
                                      row.frame_from
                                    }
                                  </td>

                                  <td>
                                    {
                                      row.frame_to
                                    }
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.displacement_um
                                    )}{" "}
                                    µm
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.radial_change_um
                                    )}{" "}
                                    µm
                                  </td>

                                  <td>
                                    {
                                      row.state
                                    }
                                  </td>

                                  <td>
                                    {hasNumber(
                                      row.speed_um_min
                                    )
                                      ? `${formatNumber(
                                          row.speed_um_min
                                        )} µm/min`
                                      : "—"}
                                  </td>

                                </tr>
                              )
                            )}

                        </tbody>

                      </table>

                    </div>
                  )}

                </>

              ) : (

                <>

                  <div className="selected-cell-stats">

                    <StatCard
                      label="AVERAGE SPEED"
                      value={formatNumber(
                        selectedCell.average_speed_px_s
                      )}
                      unit="px/s"
                    />

                    <StatCard
                      label="MAXIMUM SPEED"
                      value={formatNumber(
                        selectedCell.maximum_speed_px_s
                      )}
                      unit="px/s"
                    />

                    <StatCard
                      label="TOTAL DISTANCE"
                      value={formatNumber(
                        selectedCell.total_distance_px
                      )}
                      unit="px"
                    />

                    <StatCard
                      label="NET DISPLACEMENT"
                      value={formatNumber(
                        selectedCell.net_displacement_px
                      )}
                      unit="px"
                    />

                    <StatCard
                      label="AVERAGE AREA"
                      value={formatNumber(
                        selectedCell.average_area_px
                      )}
                      unit="px²"
                    />

                    <StatCard
                      label="MAXIMUM AREA"
                      value={formatNumber(
                        selectedCell.maximum_area_px
                      )}
                      unit="px²"
                    />

                  </div>


                  {selectedTrack.length >
                    0 && (
                    <div className="mini-track-table">

                      <div className="mini-track-title">
                        FRAME-BY-FRAME MOVEMENT
                      </div>

                      <table>

                        <thead>

                          <tr>

                            <th>
                              Frame
                            </th>

                            <th>
                              Time
                            </th>

                            <th>
                              X
                            </th>

                            <th>
                              Y
                            </th>

                            <th>
                              Area
                            </th>

                            <th>
                              Speed
                            </th>

                          </tr>

                        </thead>


                        <tbody>

                          {selectedTrack
                            .slice(
                              0,
                              60
                            )
                            .map(
                              (
                                row,
                                index
                              ) => (
                                <tr
                                  key={`${row.frame}-${index}`}
                                >

                                  <td>
                                    {
                                      row.frame
                                    }
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.time_seconds
                                    )}{" "}
                                    s
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.x ??
                                        row.centroid_x_px
                                    )}
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.y ??
                                        row.centroid_y_px
                                    )}
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.area_px ??
                                        row.area_um2
                                    )}
                                  </td>

                                  <td>
                                    {formatNumber(
                                      row.speed_px_s
                                    )}{" "}
                                    px/s
                                  </td>

                                </tr>
                              )
                            )}

                        </tbody>

                      </table>

                    </div>
                  )}

                </>

              )}

            </div>

          </section>
        )}


        {/* ====================================================
            DOWNLOADS
            ==================================================== */}

        <section className="section">

          <SectionHeading
            number={
              analysisMode ===
              "microglia"
                ? "09"
                : "08"
            }
            title="Analysis Files"
            description="All files below are taken dynamically from the current backend result instead of assuming a fixed video name."
          />


          <div className="download-grid">

            {downloadEntries.length >
            0 ? (

              downloadEntries.map(
                (
                  entry
                ) => (
                  <DownloadCard
                    key={
                      entry.key
                    }
                    title={
                      entry.title
                    }
                    description={
                      descriptionForDownload(
                        entry.key,
                        analysisMode
                      )
                    }
                    url={
                      entry.url
                    }
                    icon={
                      downloadIcon(
                        entry.key
                      )
                    }
                  />
                )
              )

            ) : (

              <div
                className="loading-card"
                style={{
                  gridColumn:
                    "1 / -1",
                }}
              >
                Analysis output files will appear here after completion.
              </div>

            )}

          </div>


          {excelUrl && (
            <div
              className="graph-download-panel"
              style={{
                marginTop: 18,
              }}
            >

              <div>

                <span>
                  SPREADSHEET PACKAGE
                </span>

                <h3>
                  Microglia analysis workbook
                </h3>

                <p>
                  Cell metrics, tracking,
                  process-tip events,
                  frame metrics and population
                  analysis.
                </p>

              </div>


              <a
                href={
                  excelUrl
                }
                download
                className="secondary-button"
              >
                Download Excel Workbook
              </a>

            </div>
          )}

        </section>


        {/* ====================================================
            FOOTER
            ==================================================== */}

        <footer className="app-footer">

          <div>

            <strong>
              CELL LAB
            </strong>

            <span>
              AI-assisted microscopy analysis
            </span>

          </div>


          <div>

            {analysisMode ===
            "microglia"
              ? "Microglia · U-Net · morphology · motility"
              : "Local processing · Instance U-Net"}

          </div>

        </footer>


      </main>

        </div>
      </div>
    </>
  );
}


// ============================================================
// SMALL COMPONENTS
// ============================================================

function VideoMetricMicroglia({
  label,
  value,
  suffix,
}) {
  return (
    <div className="video-metric">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

      <small>
        {suffix}
      </small>

    </div>
  );
}


function RankingPanelDynamic({
  title,
  subtitle,
  data,
  field,
  unit,
  onSelect,
}) {
  return (
    <div className="ranking-panel">

      <div className="ranking-header">

        <span>
          CELL RANKING
        </span>

        <h3>
          {title}
        </h3>

        <p>
          {subtitle}
        </p>

      </div>


      {data.length ? (

        data.map(
          (
            cell,
            index
          ) => {

            const id =
              cell.track_id ??
              cell.cell_id;

            return (
              <button
                key={`${id}-${index}`}
                className="ranking-button"
                onClick={() =>
                  onSelect(
                    cell
                  )
                }
              >

                <div className="ranking-row">

                  <div className="ranking-number">
                    {index + 1}
                  </div>

                  <div className="ranking-cell">

                    <span className="cell-dot" />

                    Cell #
                    {id}

                  </div>

                  <div className="ranking-value">

                    {hasNumber(
                      cell[field]
                    )
                      ? formatNumber(
                          cell[field]
                        )
                      : "—"}

                    {unit && (
                      <span>
                        {unit}
                      </span>
                    )}

                  </div>

                </div>

              </button>
            );
          }
        )

      ) : (

        <div className="ranking-empty">
          No tracking data available.
        </div>

      )}

    </div>
  );
}


// ============================================================
// GRAPH HELPERS
// ============================================================

function graphCategory(
  key
) {
  const lower =
    String(
      key || ""
    ).toLowerCase();

  if (
    lower.includes(
      "speed"
    )
  ) {
    return "Motion";
  }

  if (
    lower.includes(
      "sholl"
    )
  ) {
    return "Morphology";
  }

  if (
    lower.includes(
      "branch"
    )
  ) {
    return "Morphology";
  }

  if (
    lower.includes(
      "soma"
    )
  ) {
    return "Motility";
  }

  if (
    lower.includes(
      "tip"
    )
  ) {
    return "Motility";
  }

  if (
    lower.includes(
      "cell_count"
    ) ||
    lower.includes(
      "population"
    )
  ) {
    return "Population";
  }

  if (
    lower.includes(
      "trajectory"
    )
  ) {
    return "Tracking";
  }

  if (
    lower.includes(
      "duration"
    )
  ) {
    return "Distribution";
  }

  return "Analysis";
}


function graphDescription(
  key,
  mode
) {
  const lower =
    String(
      key || ""
    ).toLowerCase();

  if (
    lower.includes(
      "cell_count"
    )
  ) {
    return mode ===
      "microglia"
      ? "Detected microglial population across the analyzed time sequence."
      : "Detected cell population across time.";
  }

  if (
    lower.includes(
      "skeleton"
    )
  ) {
    return "Average microglial arbor / skeleton length through the analyzed sequence.";
  }

  if (
    lower.includes(
      "tip_speed"
    )
  ) {
    return "Distribution of process-tip movement speeds.";
  }

  if (
    lower.includes(
      "trajectory"
    )
  ) {
    return "Spatial trajectories of tracked objects.";
  }

  if (
    lower.includes(
      "speed"
    )
  ) {
    return "Velocity-related quantitative measurements from the current analysis.";
  }

  return "Quantitative visualization generated by the backend.";
}


function descriptionForDownload(
  key,
  mode
) {
  const lower =
    String(
      key || ""
    ).toLowerCase();

  if (
    lower.includes(
      "excel"
    )
  ) {
    return "Excel workbook containing quantitative analysis sheets.";
  }

  if (
    lower.includes(
      "track"
    )
  ) {
    return mode ===
      "microglia"
      ? "Per-frame microglia track positions and morphology."
      : "Per-frame cell tracking positions and movement.";
  }

  if (
    lower.includes(
      "tip"
    )
  ) {
    return "Process-tip transitions, extension/retraction state and speed.";
  }

  if (
    lower.includes(
      "population"
    )
  ) {
    return "Population count over time.";
  }

  if (
    lower.includes(
      "frame"
    )
  ) {
    return "Frame-level quantitative measurements.";
  }

  if (
    lower.includes(
      "report"
    )
  ) {
    return "Scientific summary and analysis report.";
  }

  if (
    lower.includes(
      "video"
    )
  ) {
    return "AI-processed microscopy output.";
  }

  return "Generated analysis output.";
}


function downloadIcon(
  key
) {
  const lower =
    String(
      key || ""
    ).toLowerCase();

  if (
    lower.includes(
      "excel"
    )
  ) {
    return "▦";
  }

  if (
    lower.includes(
      "track"
    )
  ) {
    return "⌁";
  }

  if (
    lower.includes(
      "tip"
    )
  ) {
    return "↗";
  }

  if (
    lower.includes(
      "video"
    )
  ) {
    return "▶";
  }

  if (
    lower.includes(
      "report"
    )
  ) {
    return "≡";
  }

  return "↓";
}


// ============================================================
// EXPORT
// ============================================================

export default App;