import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [stateName, setStateName] = useState("");
  const [district, setDistrict] = useState("");
  const [month, setMonth] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [currentUser, setCurrentUser] = useState(null); // { username, role }

  // auth UI state
  const [loginTab, setLoginTab] = useState("user"); // "user" | "admin"
  const [userAuthMode, setUserAuthMode] = useState("login"); // "login" | "signup"
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");

  const [deletePassword, setDeletePassword] = useState("");
  const [deleteMsg, setDeleteMsg] = useState("");

  // ---- NEW: locations for dropdowns ----
  const [locations, setLocations] = useState([]); // [{ state, state_u, districts: [...] }]
  const [districtOptions, setDistrictOptions] = useState([]);

  // ---------- GOOGLE TRANSLATE INTEGRATION (ALWAYS VISIBLE) ----------
  useEffect(() => {
    const scriptId = "google-translate-script";

    if (document.getElementById(scriptId)) return;

    window.googleTranslateElementInit = function () {
      if (!window.google || !window.google.translate) return;
      // eslint-disable-next-line no-undef
      new window.google.translate.TranslateElement(
        {
          pageLanguage: "en",
          includedLanguages: "en,hi,kn,mr,ta,te,pa,gu,bn,ml,or,as",
          layout: window.google.translate.TranslateElement.InlineLayout.SIMPLE,
        },
        "google_translate_element"
      );
    };

    const script = document.createElement("script");
    script.id = scriptId;
    script.src =
      "//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
    script.async = true;
    document.body.appendChild(script);
  }, []);
  // ---------- END GOOGLE TRANSLATE INTEGRATION ----------

  // ---------- LOAD LOCATIONS ONCE ----------
  useEffect(() => {
    async function fetchLocations() {
      try {
        const resp = await fetch("http://localhost:8000/api/locations");
        const data = await resp.json();
        if (data && Array.isArray(data.locations)) {
          setLocations(data.locations);
        } else {
          console.error("Invalid locations response", data);
        }
      } catch (err) {
        console.error("Error fetching locations:", err);
      }
    }
    fetchLocations();
  }, []);

  // whenever state changes, update district options
  useEffect(() => {
    if (!stateName) {
      setDistrictOptions([]);
      setDistrict("");
      return;
    }
    const loc = locations.find((l) => l.state === stateName);
    if (loc) {
      setDistrictOptions(loc.districts || []);
      setDistrict("");
    } else {
      setDistrictOptions([]);
      setDistrict("");
    }
  }, [stateName, locations]);

  const handlePredict = async () => {
    if (!currentUser) {
      alert("Please login (user/admin) first.");
      return;
    }
    if (!stateName || !district || !month) {
      alert("Please select state, district and enter month.");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const resp = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state: stateName,
          district,
          month,
        }),
      });
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error calling backend");
    } finally {
      setLoading(false);
    }
  };

  // ---------- AUTH HANDLERS ----------

  const handleUserLogin = async () => {
    setAuthError("");
    try {
      const resp = await fetch("http://localhost:8000/api/user/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: authUsername,
          password: authPassword,
        }),
      });
      const data = await resp.json();
      if (data.error) {
        setAuthError(data.error);
        return;
      }
      setCurrentUser({ username: data.username, role: data.role });
      setAuthPassword("");
      setAuthError("");
      setDeleteMsg("");
    } catch (err) {
      console.error(err);
      setAuthError("Error during user login.");
    }
  };

  const handleUserSignup = async () => {
    setAuthError("");
    try {
      const resp = await fetch("http://localhost:8000/api/user/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: authUsername,
          password: authPassword,
        }),
      });
      const data = await resp.json();
      if (data.error) {
        setAuthError(data.error);
        return;
      }
      setCurrentUser({ username: data.username, role: data.role });
      setAuthPassword("");
      setAuthError("");
      setDeleteMsg("");
    } catch (err) {
      console.error(err);
      setAuthError("Error during signup.");
    }
  };

  const handleAdminLogin = async () => {
    setAuthError("");
    try {
      const resp = await fetch("http://localhost:8000/api/admin/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: authUsername,
          password: authPassword,
        }),
      });
      const data = await resp.json();
      if (data.error) {
        setAuthError(data.error);
        return;
      }
      setCurrentUser({ username: data.username, role: data.role }); // admin
      setAuthPassword("");
      setAuthError("");
      setDeleteMsg("");
    } catch (err) {
      console.error(err);
      setAuthError("Error during admin login.");
    }
  };

  const handleDeleteAccount = async () => {
    if (!currentUser || currentUser.role !== "user") {
      setDeleteMsg("Only normal users can delete their account.");
      return;
    }
    if (!deletePassword) {
      setDeleteMsg("Please enter your password to confirm.");
      return;
    }
    try {
      const resp = await fetch("http://localhost:8000/api/user/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: currentUser.username,
          password: deletePassword,
        }),
      });
      const data = await resp.json();
      if (data.error) {
        setDeleteMsg(data.error);
      } else {
        setDeleteMsg(data.message || "Account deleted.");
        setCurrentUser(null);
        setResult(null);
        setStateName("");
        setDistrict("");
        setMonth("");
        setDeletePassword("");
      }
    } catch (err) {
      console.error(err);
      setDeleteMsg("Error deleting account.");
    }
  };

  const fmt = (v) =>
    typeof v === "number"
      ? v.toLocaleString("en-IN", { maximumFractionDigits: 0 })
      : v;

  return (
    <div className="app-root">
      {/* 🌐 GOOGLE TRANSLATE (ALWAYS VISIBLE) */}
      <div id="google_translate_element" className="google-translate" />

      {/* AUTH SECTION (shown when not logged in) */}
      {!currentUser && (
        <section className="card auth-card">
          <h2 className="card-title">Login / Signup</h2>

          {/* Tabs: User / Admin */}
          <div className="tabs">
            <button
              type="button"
              onClick={() => {
                setLoginTab("user");
                setUserAuthMode("login");
                setAuthError("");
              }}
              className={`tab-button ${loginTab === "user" ? "active" : ""}`}
            >
              User
            </button>
            <button
              type="button"
              onClick={() => {
                setLoginTab("admin");
                setAuthError("");
              }}
              className={`tab-button ${loginTab === "admin" ? "active" : ""}`}
            >
              Admin
            </button>
          </div>

          {/* Common username/password fields */}
          <div className="auth-form">
            <input
              className="text-input"
              placeholder={loginTab === "admin" ? "Admin username" : "Username"}
              value={authUsername}
              onChange={(e) => setAuthUsername(e.target.value)}
            />
            <input
              className="text-input"
              placeholder="Password"
              type="password"
              value={authPassword}
              onChange={(e) => setAuthPassword(e.target.value)}
            />

            {authError && <p className="error-text">{authError}</p>}

            {/* User login/signup */}
            {loginTab === "user" && (
              <>
                {userAuthMode === "login" ? (
                  <button
                    onClick={handleUserLogin}
                    className="btn btn-primary full-width"
                    disabled={!authUsername || !authPassword}
                  >
                    User Login
                  </button>
                ) : (
                  <button
                    onClick={handleUserSignup}
                    className="btn btn-primary full-width"
                    disabled={!authUsername || !authPassword}
                  >
                    Create User Account
                  </button>
                )}

                <button
                  type="button"
                  className="link-button"
                  onClick={() =>
                    setUserAuthMode((m) => (m === "login" ? "signup" : "login"))
                  }
                >
                  {userAuthMode === "login"
                    ? "New user? Create an account"
                    : "Already have an account? Login"}
                </button>
              </>
            )}

            {/* Admin login */}
            {loginTab === "admin" && (
              <>
                <button
                  onClick={handleAdminLogin}
                  className="btn btn-primary full-width"
                  disabled={!authUsername || !authPassword}
                >
                  Admin Login
                </button>
                <p className="small-muted">
                  Default admin: <strong>Saksham / 1234</strong>
                </p>
              </>
            )}
          </div>
        </section>
      )}

      {/* Message when not logged in */}
      {!currentUser && (
        <p className="login-hint">
          Please login as <strong>User</strong> or <strong>Admin</strong> to
          access the crop predictor.
        </p>
      )}

      {/* TOP BAR AFTER LOGIN */}
      {currentUser && (
        <section className="card topbar">
          <div className="topbar-user">
            <span className="status-dot" />
            <span>
              Logged in as{" "}
              <strong>
                {currentUser.username} ({currentUser.role})
              </strong>
            </span>
          </div>
          <div className="topbar-actions">
            {currentUser.role === "user" && (
              <>
                <input
                  type="password"
                  placeholder="Password to delete"
                  value={deletePassword}
                  onChange={(e) => setDeletePassword(e.target.value)}
                  className="text-input text-input-compact"
                />
                <button
                  onClick={handleDeleteAccount}
                  className="btn btn-danger"
                >
                  Delete my account
                </button>
              </>
            )}
            <button
              onClick={() => {
                setCurrentUser(null);
                setResult(null);
                setStateName("");
                setDistrict("");
                setMonth("");
                setDeletePassword("");
                setDeleteMsg("");
              }}
              className="btn btn-outline"
            >
              Logout
            </button>
          </div>
          {deleteMsg && <p className="delete-msg">{deleteMsg}</p>}
        </section>
      )}

      {/* PREDICTOR ONLY AFTER LOGIN */}
      {currentUser && (
        <>
          <section className="card predictor-card">
            <div className="predictor-header">
              <div>
                <h1 className="app-title">
                  Crop Planner{" "}
                  <span className="app-title-pill">Top 3 by profit</span>
                </h1>
                <p className="app-subtitle">
                  Select your state and district to get smart crop
                  recommendations based on cost, profit, weather and risk.
                </p>
              </div>
              <div className="hero-illustration" />
            </div>

            {/* Inputs */}
            <div className="predictor-form">
              {/* STATE DROPDOWN */}
              <select
                className="text-input"
                value={stateName}
                onChange={(e) => setStateName(e.target.value)}
              >
                <option value="">Select State</option>
                {locations.map((loc) => (
                  <option key={loc.state_u} value={loc.state}>
                    {loc.state}
                  </option>
                ))}
              </select>

              {/* DISTRICT DROPDOWN */}
              <select
                className="text-input"
                value={district}
                onChange={(e) => setDistrict(e.target.value)}
                disabled={!stateName || districtOptions.length === 0}
              >
                <option value="">
                  {stateName ? "Select District" : "Select state first"}
                </option>
                {districtOptions.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>

              {/* MONTH input (you can also convert this to dropdown if you want) */}
              <input
                className="text-input"
                placeholder="Month (e.g. July)"
                value={month}
                onChange={(e) => setMonth(e.target.value)}
              />

              <button
                onClick={handlePredict}
                disabled={
                  loading || !stateName || !district || !month
                }
                className="btn btn-primary full-width"
              >
                {loading ? "Calculating..." : "Get Recommendations"}
              </button>
            </div>
          </section>

          {/* Error */}
          {result && result.error && (
            <p className="error-text error-text-inline">{result.error}</p>
          )}

          {/* Results */}
          {result && !result.error && (
            <div className="results-grid">
              {/* CROPS SECTION */}
              <section className="card result-card">
                <h2 className="card-title">Top 3 Crops</h2>
                <p className="meta-line">
                  <strong>State:</strong> {result.state} &nbsp;|&nbsp;
                  <strong>District:</strong> {result.district} &nbsp;|&nbsp;
                  <strong>Month:</strong> {result.month}
                  {result.season && (
                    <>
                      {" "}
                      &nbsp;|&nbsp; <strong>Season:</strong> {result.season}
                    </>
                  )}
                </p>

                <ul className="crop-list">
                  {result.best_crops?.map((c) => {
                    const profitAdj = c.profit_per_ha ?? 0;
                    const profitBase =
                      c.base_profit_per_ha ?? c.profit_for_1_ha ?? profitAdj;

                    const cost = c.cost_per_ha ?? null;
                    const market = c.market_price ?? null;

                    const revenue = cost != null ? profitAdj + cost : null;
                    const margin =
                      cost != null && revenue > 0
                        ? (profitAdj / revenue) * 100
                        : null;

                    return (
                      <li key={c.crop} className="crop-item">
                        <div className="crop-header">
                          <div className="crop-name">{c.crop}</div>
                          <div className="chip chip-success">
                            ₹{fmt(profitAdj)}/ha (adj.)
                          </div>
                        </div>

                        <div className="crop-row">
                          <span>💰 Adjusted profit (this month)</span>
                          <span>₹{fmt(profitAdj)}/ha</span>
                        </div>

                        <div className="crop-row">
                          <span>📘 Base historical / model profit</span>
                          <span>₹{fmt(profitBase)}/ha</span>
                        </div>

                        {cost != null && (
                          <div className="crop-row">
                            <span>🧾 Cost of cultivation</span>
                            <span>₹{fmt(cost)}/ha</span>
                          </div>
                        )}

                        {market != null && (
                          <div className="crop-row">
                            <span>🏷 Predicted market price (Rs per quintal)</span>
                            <span>₹{fmt(market)}</span>
                          </div>
                        )}

                        {revenue != null && (
                          <div className="crop-row">
                            <span>📈 Expected value (cost + adj. profit)</span>
                            <span>₹{fmt(revenue)}/ha</span>
                          </div>
                        )}

                        {margin != null && (
                          <div className="crop-row">
                            <span>📊 Profit margin (this month)</span>
                            <span>{margin.toFixed(1)}%</span>
                          </div>
                        )}

                        <div className="crop-row">
                          <span>⏱ Growth time</span>
                          <span>
                            {c.growth_days} days (~{c.growth_months} months)
                          </span>
                        </div>

                        {c.irrigation && (
                          <div className="crop-note">
                            🚿 <strong>Irrigation:</strong> {c.irrigation}
                          </div>
                        )}
                        {c.fertilizers && (
                          <div className="crop-note">
                            🧪 <strong>Fertilizers:</strong> {c.fertilizers}
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </section>

              {/* WEATHER */}
              <section className="card result-card">
                <h2 className="card-title">Live Weather &amp; Water</h2>
                <p className="meta-line">
                  <strong>Lat:</strong>{" "}
                  {result.lat != null ? result.lat.toFixed(3) : "N/A"}
                  &nbsp;|&nbsp;
                  <strong>Lon:</strong>{" "}
                  {result.lon != null ? result.lon.toFixed(3) : "N/A"}
                </p>
                <div className="stat-grid">
                  <div className="stat-item">
                    <span className="stat-label">Temperature</span>
                    <span className="stat-value">
                      {result.weather?.temp ?? "N/A"} °C
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Humidity</span>
                    <span className="stat-value">
                      {result.weather?.humidity ?? "N/A"} %
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Pressure</span>
                    <span className="stat-value">
                      {result.weather?.pressure ?? "N/A"} hPa
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Precipitation</span>
                    <span className="stat-value">
                      {result.weather?.precip_mm ?? "N/A"} mm
                    </span>
                  </div>
                  <div className="stat-item stat-wide">
                    <span className="stat-label">
                      River discharge (approx water level)
                    </span>
                    <span className="stat-value">
                      {result.river_discharge_m3s != null
                        ? `${result.river_discharge_m3s.toFixed(1)} m³/s`
                        : "N/A"}
                    </span>
                  </div>
                </div>
              </section>

              {/* RISK */}
              <section className="card result-card">
                <h2 className="card-title">Risk &amp; Warning</h2>
                <div className="risk-bars">
                  <div className="risk-row">
                    <span>Normal</span>
                    <div className="risk-bar">
                      <div
                        className="risk-bar-fill normal"
                        style={{ width: `${result.risk?.normal || 0}%` }}
                      />
                    </div>
                    <span className="risk-value">
                      {result.risk?.normal}%
                    </span>
                  </div>
                  <div className="risk-row">
                    <span>Flood</span>
                    <div className="risk-bar">
                      <div
                        className="risk-bar-fill flood"
                        style={{ width: `${result.risk?.flood || 0}%` }}
                      />
                    </div>
                    <span className="risk-value">
                      {result.risk?.flood}%
                    </span>
                  </div>
                  <div className="risk-row">
                    <span>Drought</span>
                    <div className="risk-bar">
                      <div
                        className="risk-bar-fill drought"
                        style={{ width: `${result.risk?.drought || 0}%` }}
                      />
                    </div>
                    <span className="risk-value">
                      {result.risk?.drought}%
                    </span>
                  </div>
                </div>
                <p className="warning-text">
                  <strong>Warning:</strong> {result.risk?.warning}
                </p>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;
