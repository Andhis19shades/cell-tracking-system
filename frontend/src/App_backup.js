import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStatus, setAnalysisStatus] = useState("");
  const [results, setResults] = useState(null);
  const [activeSection, setActiveSection] = useState("upload");

  const fileInputRef = useRef(null);

  const [settings, setSettings] = useState({
    fps: 8,
    pixelSize: "",
    tracking: true,
    segmentation: "instance",
    opticalFlow: true,
    watershed: true,
  });

  const supportedVideo = [
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
  ];

  const supportedImage = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
  ];

  const isSupportedFile = (file) => {
    const name = file.name.toLowerCase();

    return (
      supportedVideo.some((ext) => name.endsWith(ext)) ||
      supportedImage.some((ext) => name.endsWith(ext))
    );
  };

  const addFiles = (newFiles) => {
    const validFiles = Array.from(newFiles).filter(isSupportedFile);

    if (!validFiles.length) {
      alert(
        "Please upload a microscopy video or supported image file."
      );
      return;
    }

    setFiles((previous) => {
      const existing = new Set(
        previous.map((file) => file.name)
      );

      const unique = validFiles.filter(
        (file) => !existing.has(file.name)
      );

      return [...previous, ...unique];
    });

    setResults(null);
  };

  const handleFileInput = (event) => {
    addFiles(event.target.files);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    addFiles(event.dataTransfer.files);
  };

  const removeFile = (index) => {
    setFiles((previous) =>
      previous.filter((_, i) => i !== index)
    );
  };

  const clearFiles = () => {
    setFiles([]);
    setResults(null);
    setAnalysisProgress(0);
    setAnalysisStatus("");
  };

  const updateSetting = (key, value) => {
    setSettings((previous) => ({
      ...previous,
      [key]: value,
    }));
  };

  /*
   * ---------------------------------------------------------
   * DEMO RESULTS
   * ---------------------------------------------------------
   *
   * These values are only used until the Python backend is
   * connected.
   *
   * Once the backend exists, the real values will replace
   * these automatically.
   */

  const demoResults = useMemo(
    () => ({
      cells: 431,
      movingCells: 82,
      meanSpeed: 52.88,
      medianSpeed: 31.42,
      maxSpeed: 639.99,
      meanDisplacement: 18.73,
      totalMovement: 8062.4,
      meanArea: 251.67,
      tracks: 11714,
      duration: 100,
      frames: 800,
      fps: 8,
    }),
    []
  );

  /*
   * ---------------------------------------------------------
   * RUN ANALYSIS
   * ---------------------------------------------------------
   */

  const runAnalysis = async () => {
    if (!files.length) {
      alert("Please upload a video or image first.");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setAnalysisStatus("Preparing microscopy data...");
    setActiveSection("results");

    /*
     * The frontend is prepared for the real Python backend.
     *
     * The backend endpoint will be:
     *
     * POST http://127.0.0.1:8000/analyze
     *
     * For now, if the backend isn't available, we show a
     * realistic processing simulation so the UI can be
     * developed independently.
     */

    try {
      const formData = new FormData();

      files.forEach((file) => {
        formData.append("files", file);
      });

      formData.append(
        "settings",
        JSON.stringify(settings)
      );

      setAnalysisStatus("Uploading data...");
      setAnalysisProgress(10);

      try {
        const response = await fetch(
          `${API_URL}/analyze`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (response.ok) {
          const data = await response.json();

          setResults(data);
          setAnalysisProgress(100);
          setAnalysisStatus(
            "Analysis completed successfully."
          );
          setIsAnalyzing(false);
          return;
        }
      } catch (backendError) {
        console.log(
          "Backend not connected yet. Using frontend preview."
        );
      }

      /*
       * Temporary frontend preview.
       */

      const stages = [
        [20, "Loading microscopy frames..."],
        [35, "Running AI segmentation..."],
        [50, "Separating individual cells..."],
        [65, "Tracking cells across frames..."],
        [78, "Calculating cell movement..."],
        [90, "Generating quantitative results..."],
        [100, "Analysis completed."],
      ];

      for (const [progress, status] of stages) {
        await new Promise((resolve) =>
          setTimeout(resolve, 650)
        );

        setAnalysisProgress(progress);
        setAnalysisStatus(status);
      }

      setResults(demoResults);
    } finally {
      setIsAnalyzing(false);
    }
  };

  /*
   * ---------------------------------------------------------
   * FORMAT HELPERS
   * ---------------------------------------------------------
   */

  const formatBytes = (bytes) => {
    if (!bytes) return "0 KB";

    const mb = bytes / (1024 * 1024);

    if (mb >= 1) {
      return `${mb.toFixed(2)} MB`;
    }

    return `${(bytes / 1024).toFixed(1)} KB`;
  };

  const formatNumber = (number, digits = 2) => {
    if (number === undefined || number === null) {
      return "—";
    }

    return Number(number).toLocaleString(undefined, {
      maximumFractionDigits: digits,
    });
  };

  /*
   * ---------------------------------------------------------
   * RESULT CARD
   * ---------------------------------------------------------
   */

  const StatCard = ({
    icon,
    label,
    value,
    unit,
    accent = "",
  }) => (
    <div className={`stat-card ${accent}`}>
      <div className="stat-icon">{icon}</div>

      <div className="stat-content">
        <div className="stat-label">{label}</div>

        <div className="stat-value">
          {value}
          {unit && (
            <span className="stat-unit">
              {unit}
            </span>
          )}
        </div>
      </div>
    </div>
  );

  /*
   * ---------------------------------------------------------
   * NAVIGATION
   * ---------------------------------------------------------
   */

  const scrollTo = (section) => {
    setActiveSection(section);

    const element = document.getElementById(section);

    if (element) {
      element.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  /*
   * ---------------------------------------------------------
   * UI
   * ---------------------------------------------------------
   */

  return (
    <div className="app-shell">

      {/* ====================================================
          SIDEBAR
      ==================================================== */}

      <aside className="sidebar">

        <div className="brand">
          <div className="brand-mark">
            <span>✦</span>
          </div>

          <div>
            <div className="brand-title">
              CELL LAB
            </div>

            <div className="brand-subtitle">
              AI MICROSCOPE ANALYSIS
            </div>
          </div>
        </div>

        <nav className="side-nav">

          <button
            className={
              activeSection === "upload"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => scrollTo("upload")}
          >
            <span>＋</span>
            Upload Data
          </button>

          <button
            className={
              activeSection === "settings"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => scrollTo("settings")}
          >
            <span>⚙</span>
            Analysis Settings
          </button>

          <button
            className={
              activeSection === "results"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => scrollTo("results")}
          >
            <span>▦</span>
            Results
          </button>

          <button
            className={
              activeSection === "tracks"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => scrollTo("tracks")}
          >
            <span>⌁</span>
            Cell Tracking
          </button>

        </nav>

        <div className="sidebar-bottom">

          <div className="local-status">
            <span className="status-dot"></span>

            <div>
              <strong>LOCAL AI</strong>
              <small>
                Processing on this computer
              </small>
            </div>
          </div>

          <div className="model-info">
            <span>MODEL</span>
            <strong>Instance U-Net</strong>
          </div>

        </div>

      </aside>


      {/* ====================================================
          MAIN
      ==================================================== */}

      <main className="main-content">

        {/* TOP BAR */}

        <header className="topbar">

          <div>
            <div className="eyebrow">
              MICROSCOPY / ANALYSIS WORKSPACE
            </div>

            <h1>
              Cell Analysis Lab
            </h1>
          </div>

          <div className="topbar-right">

            <div className="connection-pill">
              <span className="status-dot"></span>
              Localhost
            </div>

            <div className="version">
              v1.0
            </div>

          </div>

        </header>


        {/* ==================================================
            UPLOAD
        ================================================== */}

        <section
          id="upload"
          className="section"
        >

          <div className="section-heading">

            <div>
              <div className="section-number">
                01
              </div>

              <h2>
                Upload Microscopy Data
              </h2>

              <p>
                Upload videos or microscopy images.
                The AI pipeline will handle segmentation,
                cell separation, tracking and motion analysis.
              </p>
            </div>

          </div>


          <div
            className={
              isDragging
                ? "upload-zone dragging"
                : "upload-zone"
            }
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() =>
              fileInputRef.current?.click()
            }
          >

            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".avi,.mp4,.mov,.mkv,.png,.jpg,.jpeg,.tif,.tiff"
              onChange={handleFileInput}
              hidden
            />

            <div className="upload-icon">
              ↑
            </div>

            <h3>
              Drop your microscopy files here
            </h3>

            <p>
              or click to browse your computer
            </p>

            <div className="file-types">
              VIDEO
              <span>AVI</span>
              <span>MP4</span>
              <span>MOV</span>
              <span>MKV</span>

              <i></i>

              IMAGE
              <span>TIFF</span>
              <span>PNG</span>
              <span>JPG</span>
            </div>

          </div>


          {/* FILE LIST */}

          {files.length > 0 && (
            <div className="uploaded-files">

              <div className="file-list-header">
                <div>
                  Uploaded Data
                  <span className="count-badge">
                    {files.length}
                  </span>
                </div>

                <button
                  onClick={clearFiles}
                  className="clear-button"
                >
                  Clear all
                </button>
              </div>

              {files.map((file, index) => {

                const isVideo =
                  /\.(avi|mp4|mov|mkv)$/i.test(
                    file.name
                  );

                return (
                  <div
                    className="file-row"
                    key={`${file.name}-${index}`}
                  >

                    <div className="file-type-icon">
                      {isVideo ? "▶" : "▧"}
                    </div>

                    <div className="file-info">

                      <strong>
                        {file.name}
                      </strong>

                      <span>
                        {isVideo
                          ? "Microscopy video"
                          : "Microscopy image"}
                        {" · "}
                        {formatBytes(file.size)}
                      </span>

                    </div>

                    <div className="file-ready">
                      READY
                    </div>

                    <button
                      className="remove-file"
                      onClick={(event) => {
                        event.stopPropagation();
                        removeFile(index);
                      }}
                    >
                      ×
                    </button>

                  </div>
                );
              })}

            </div>
          )}

        </section>


        {/* ==================================================
            SETTINGS
        ================================================== */}

        <section
          id="settings"
          className="section"
        >

          <div className="section-heading">

            <div>
              <div className="section-number">
                02
              </div>

              <h2>
                Analysis Settings
              </h2>

              <p>
                Configure the microscope parameters used
                for quantitative motion analysis.
              </p>
            </div>

          </div>


          <div className="settings-grid">

            {/* FPS */}

            <div className="setting-card">

              <label>
                Frame Rate
              </label>

              <div className="input-with-unit">

                <input
                  type="number"
                  min="1"
                  value={settings.fps}
                  onChange={(event) =>
                    updateSetting(
                      "fps",
                      event.target.value
                    )
                  }
                />

                <span>
                  FPS
                </span>

              </div>

              <small>
                Frames captured per second
              </small>

            </div>


            {/* PIXEL SIZE */}

            <div className="setting-card">

              <label>
                Pixel Calibration
              </label>

              <div className="input-with-unit">

                <input
                  type="number"
                  step="0.01"
                  placeholder="Auto"
                  value={settings.pixelSize}
                  onChange={(event) =>
                    updateSetting(
                      "pixelSize",
                      event.target.value
                    )
                  }
                />

                <span>
                  μm/px
                </span>

              </div>

              <small>
                Optional microscope calibration
              </small>

            </div>


            {/* SEGMENTATION */}

            <div className="setting-card">

              <label>
                Segmentation
              </label>

              <select
                value={settings.segmentation}
                onChange={(event) =>
                  updateSetting(
                    "segmentation",
                    event.target.value
                  )
                }
              >

                <option value="instance">
                  Instance-aware
                </option>

                <option value="binary">
                  Binary
                </option>

              </select>

              <small>
                Cell segmentation method
              </small>

            </div>


            {/* TRACKING */}

            <div className="setting-card">

              <label>
                Cell Tracking
              </label>

              <button
                className={
                  settings.tracking
                    ? "toggle active"
                    : "toggle"
                }
                onClick={() =>
                  updateSetting(
                    "tracking",
                    !settings.tracking
                  )
                }
              >

                <span></span>

                <strong>
                  {settings.tracking
                    ? "Enabled"
                    : "Disabled"}
                </strong>

              </button>

              <small>
                Track cells across frames
              </small>

            </div>


            {/* WATERSHED */}

            <div className="setting-card">

              <label>
                Cell Separation
              </label>

              <button
                className={
                  settings.watershed
                    ? "toggle active"
                    : "toggle"
                }
                onClick={() =>
                  updateSetting(
                    "watershed",
                    !settings.watershed
                  )
                }
              >

                <span></span>

                <strong>
                  {settings.watershed
                    ? "Watershed ON"
                    : "Watershed OFF"}
                </strong>

              </button>

              <small>
                Separate touching cells
              </small>

            </div>


            {/* OPTICAL FLOW */}

            <div className="setting-card">

              <label>
                Temporal Analysis
              </label>

              <button
                className={
                  settings.opticalFlow
                    ? "toggle active"
                    : "toggle"
                }
                onClick={() =>
                  updateSetting(
                    "opticalFlow",
                    !settings.opticalFlow
                  )
                }
              >

                <span></span>

                <strong>
                  {settings.opticalFlow
                    ? "Enabled"
                    : "Disabled"}
                </strong>

              </button>

              <small>
                Temporal motion estimation
              </small>

            </div>

          </div>


          {/* RUN BUTTON */}

          <div className="run-analysis-container">

            <button
              className={
                isAnalyzing
                  ? "run-button analyzing"
                  : "run-button"
              }
              onClick={runAnalysis}
              disabled={isAnalyzing}
            >

              {isAnalyzing ? (
                <>
                  <span className="spinner"></span>

                  ANALYZING
                  {" "}
                  {analysisProgress}%
                </>
              ) : (
                <>
                  RUN AI ANALYSIS
                  <span>→</span>
                </>
              )}

            </button>

            {isAnalyzing && (
              <div className="progress-area">

                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${analysisProgress}%`,
                    }}
                  />
                </div>

                <span>
                  {analysisStatus}
                </span>

              </div>
            )}

          </div>

        </section>


        {/* ==================================================
            RESULTS
        ================================================== */}

        <section
          id="results"
          className="section results-section"
        >

          <div className="section-heading">

            <div>
              <div className="section-number">
                03
              </div>

              <h2>
                Quantitative Results
              </h2>

              <p>
                Numerical measurements extracted from
                the analyzed microscopy data.
              </p>
            </div>

            {results && (
              <div className="analysis-complete">
                ● ANALYSIS COMPLETE
              </div>
            )}

          </div>


          {!results && !isAnalyzing && (
            <div className="empty-results">

              <div className="empty-icon">
                ◇
              </div>

              <h3>
                No analysis yet
              </h3>

              <p>
                Upload microscopy data and run the AI
                analysis to see quantitative results here.
              </p>

            </div>
          )}


          {isAnalyzing && (
            <div className="processing-card">

              <div className="processing-animation">
                <div></div>
                <div></div>
                <div></div>
              </div>

              <h3>
                AI is analyzing your data
              </h3>

              <p>
                {analysisStatus}
              </p>

              <div className="large-progress">
                <div
                  style={{
                    width: `${analysisProgress}%`,
                  }}
                />
              </div>

              <strong>
                {analysisProgress}%
              </strong>

            </div>
          )}


          {results && !isAnalyzing && (

            <>

              <div className="stats-grid">

                <StatCard
                  icon="◉"
                  label="Average Cells / Frame"
                  value={formatNumber(
                    results.cells
                  )}
                  accent="green"
                />

                <StatCard
                  icon="↗"
                  label="Moving Cells / Frame"
                  value={formatNumber(
                    results.movingCells
                  )}
                  accent="blue"
                />

                <StatCard
                  icon="≈"
                  label="Mean Speed"
                  value={formatNumber(
                    results.meanSpeed
                  )}
                  unit="px/s"
                  accent="purple"
                />

                <StatCard
                  icon="⚡"
                  label="Maximum Speed"
                  value={formatNumber(
                    results.maxSpeed
                  )}
                  unit="px/s"
                  accent="orange"
                />

                <StatCard
                  icon="↔"
                  label="Mean Displacement"
                  value={formatNumber(
                    results.meanDisplacement
                  )}
                  unit="px"
                />

                <StatCard
                  icon="∑"
                  label="Total Movement"
                  value={formatNumber(
                    results.totalMovement
                  )}
                  unit="px"
                />

                <StatCard
                  icon="□"
                  label="Mean Cell Area"
                  value={formatNumber(
                    results.meanArea
                  )}
                  unit="px²"
                />

                <StatCard
                  icon="⌁"
                  label="Tracked Objects"
                  value={formatNumber(
                    results.tracks,
                    0
                  )}
                />

              </div>


              {/* VIDEO + CHART */}

              <div className="analysis-grid">

                <div className="panel video-panel">

                  <div className="panel-header">

                    <div>
                      <span className="panel-kicker">
                        OUTPUT
                      </span>

                      <h3>
                        Analyzed Video
                      </h3>
                    </div>

                    <div className="live-pill">
                      AI
                    </div>

                  </div>

                  <div className="video-placeholder">

                    <div className="microscope-grid"></div>

                    <div className="cell-demo cell-one">
                      <span>23</span>
                    </div>

                    <div className="cell-demo cell-two">
                      <span>41</span>
                    </div>

                    <div className="cell-demo cell-three">
                      <span>57</span>
                    </div>

                    <div className="movement-arrow arrow-one">
                      ↗
                    </div>

                    <div className="movement-arrow arrow-two">
                      →
                    </div>

                    <div className="video-overlay">

                      <span>
                        FRAME 423
                      </span>

                      <span>
                        CELLS 428
                      </span>

                      <span>
                        MOVING 79
                      </span>

                    </div>

                  </div>

                  <div className="video-footer">

                    <span>
                      Segmentation + Tracking Overlay
                    </span>

                    <span>
                      {results.fps || settings.fps} FPS
                    </span>

                  </div>

                </div>


                <div className="panel chart-panel">

                  <div className="panel-header">

                    <div>
                      <span className="panel-kicker">
                        TEMPORAL ANALYSIS
                      </span>

                      <h3>
                        Cell Activity
                      </h3>
                    </div>

                    <span className="chart-unit">
                      px/s
                    </span>

                  </div>

                  <div className="chart">

                    <div className="y-label y1">
                      100
                    </div>

                    <div className="y-label y2">
                      75
                    </div>

                    <div className="y-label y3">
                      50
                    </div>

                    <div className="y-label y4">
                      25
                    </div>

                    <svg
                      viewBox="0 0 600 300"
                      preserveAspectRatio="none"
                    >

                      <defs>

                        <linearGradient
                          id="chartGradient"
                          x1="0"
                          x2="0"
                          y1="0"
                          y2="1"
                        >

                          <stop
                            offset="0%"
                            stopColor="#39e6ba"
                            stopOpacity="0.35"
                          />

                          <stop
                            offset="100%"
                            stopColor="#39e6ba"
                            stopOpacity="0"
                          />

                        </linearGradient>

                      </defs>

                      <path
                        d="
                          M0 235
                          C30 220 45 190 70 205
                          C95 220 105 145 135 165
                          C160 185 170 110 195 130
                          C220 150 230 175 255 140
                          C280 105 290 125 315 95
                          C340 65 350 120 375 100
                          C400 80 420 125 445 90
                          C470 55 485 75 510 105
                          C535 135 550 70 575 80
                          C590 85 600 60 600 60
                          L600 300
                          L0 300
                          Z
                        "
                        fill="url(#chartGradient)"
                      />

                      <path
                        d="
                          M0 235
                          C30 220 45 190 70 205
                          C95 220 105 145 135 165
                          C160 185 170 110 195 130
                          C220 150 230 175 255 140
                          C280 105 290 125 315 95
                          C340 65 350 120 375 100
                          C400 80 420 125 445 90
                          C470 55 485 75 510 105
                          C535 135 550 70 575 80
                          C590 85 600 60 600 60
                        "
                        fill="none"
                        stroke="#16cda4"
                        strokeWidth="4"
                      />

                    </svg>

                    <div className="x-axis">
                      <span>0s</span>
                      <span>20s</span>
                      <span>40s</span>
                      <span>60s</span>
                      <span>80s</span>
                      <span>100s</span>
                    </div>

                  </div>

                  <div className="chart-summary">

                    <div>
                      <span>Mean</span>
                      <strong>
                        {formatNumber(
                          results.meanSpeed
                        )} px/s
                      </strong>
                    </div>

                    <div>
                      <span>Median</span>
                      <strong>
                        {formatNumber(
                          results.medianSpeed
                        )} px/s
                      </strong>
                    </div>

                    <div>
                      <span>Maximum</span>
                      <strong>
                        {formatNumber(
                          results.maxSpeed
                        )} px/s
                      </strong>
                    </div>

                  </div>

                </div>

              </div>


              {/* DOWNLOADS */}

              <div className="download-row">

                <button className="download-button">
                  <span>↓</span>
                  Download CSV
                </button>

                <button className="download-button secondary">
                  <span>↓</span>
                  Download Report
                </button>

                <button className="download-button secondary">
                  <span>↓</span>
                  Download Analyzed Video
                </button>

              </div>

            </>

          )}

        </section>


        {/* ==================================================
            TRACKS
        ================================================== */}

        <section
          id="tracks"
          className="section"
        >

          <div className="section-heading">

            <div>
              <div className="section-number">
                04
              </div>

              <h2>
                Cell Tracking
              </h2>

              <p>
                Individual cell movement and trajectory
                information extracted from the video.
              </p>
            </div>

          </div>


          {results ? (

            <div className="tracking-panel">

              <div className="tracking-header">

                <div>
                  <span>
                    INDIVIDUAL CELL DATA
                  </span>

                  <h3>
                    Top tracked cells
                  </h3>
                </div>

                <div className="tracking-count">
                  {results.tracks || 0} tracks
                </div>

              </div>


              <div className="track-table">

                <div className="track-row table-head">

                  <span>
                    CELL
                  </span>

                  <span>
                    FRAMES
                  </span>

                  <span>
                    AVG SPEED
                  </span>

                  <span>
                    MAX SPEED
                  </span>

                  <span>
                    DISTANCE
                  </span>

                  <span>
                    AREA
                  </span>

                </div>


                {[
                  {
                    id: 23,
                    frames: 94,
                    avg: 4.82,
                    max: 12.4,
                    distance: 182,
                    area: 247,
                  },
                  {
                    id: 41,
                    frames: 91,
                    avg: 8.14,
                    max: 17.2,
                    distance: 319,
                    area: 281,
                  },
                  {
                    id: 57,
                    frames: 86,
                    avg: 2.93,
                    max: 9.1,
                    distance: 104,
                    area: 231,
                  },
                  {
                    id: 81,
                    frames: 77,
                    avg: 11.32,
                    max: 22.7,
                    distance: 442,
                    area: 304,
                  },
                  {
                    id: 102,
                    frames: 73,
                    avg: 6.42,
                    max: 14.8,
                    distance: 215,
                    area: 266,
                  },
                ].map((cell) => (

                  <div
                    className="track-row"
                    key={cell.id}
                  >

                    <span className="cell-id">
                      <i></i>
                      #{cell.id}
                    </span>

                    <span>
                      {cell.frames}
                    </span>

                    <span className="speed-value">
                      {cell.avg} px/s
                    </span>

                    <span>
                      {cell.max} px/s
                    </span>

                    <span>
                      {cell.distance} px
                    </span>

                    <span>
                      {cell.area} px²
                    </span>

                  </div>

                ))}

              </div>

            </div>

          ) : (

            <div className="empty-tracks">

              <span>
                ⌁
              </span>

              <p>
                Cell tracking results will appear here
                after analysis.
              </p>

            </div>

          )}

        </section>


        {/* ==================================================
            FOOTER
        ================================================== */}

        <footer>

          <div>
            CELL ANALYSIS LAB
          </div>

          <span>
            Deep Learning · Segmentation · Tracking · Motion Analysis
          </span>

          <div className="footer-local">
            ● Running locally
          </div>

        </footer>

      </main>

    </div>
  );
}

export default App;
