import { useEffect, useMemo, useState } from "react";

type Verdict = "supported" | "contradicted" | "not_found";
type Evidence = { file: string; start_line: number; end_line: number; text: string };
type Claim = {
  claim: string;
  claim_type: string;
  verdict: Verdict;
  reasoning: string;
  evidence: Evidence[];
};
type Repo = { name: string; repo: string; has_report: boolean };
type Report = { repo: string; paper: string; generated_at: string; claims: Claim[] };

const labels: Record<Verdict, string> = {
  supported: "Supported",
  contradicted: "Contradicted",
  not_found: "Not found",
};

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export default function App() {
  const [repos, setRepos] = useState<Repo[]>([]);
  const [repoName, setRepoName] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState("");
  const [loadingRepos, setLoadingRepos] = useState(true);
  const [loadingReport, setLoadingReport] = useState(false);

  useEffect(() => {
    requestJson<Repo[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        const first = items.find((item) => item.has_report);
        if (first) setRepoName(first.name);
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoadingRepos(false));
  }, []);

  useEffect(() => {
    if (!repoName || !repos.find((item) => item.name === repoName)?.has_report) {
      setReport(null);
      return;
    }
    setLoadingReport(true);
    setError("");
    requestJson<Report>(`/api/report/${encodeURIComponent(repoName)}`)
      .then((data) => {
        setReport(data);
        setSelected(0);
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoadingReport(false));
  }, [repoName, repos]);

  const claim = report?.claims[selected];
  const counts = useMemo(() => {
    const base = { supported: 0, contradicted: 0, not_found: 0 };
    return (report?.claims || []).reduce((total, item) => {
      total[item.verdict] += 1;
      return total;
    }, base);
  }, [report]);

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="wordmark" href="/" aria-label="ReproLens home">
          <span className="mark">R</span>
          <span>repro<span className="wordmark-light">lens</span></span>
        </a>
        <div className="topbar-note"><span className="live-dot" /> PAPER ↔ REPOSITORY CHECK</div>
      </header>

      <section className="hero">
        <div className="eyebrow">RESEARCH REPRODUCIBILITY · EVIDENCE FIRST</div>
        <h1>Does the code<br /><span>match the paper?</span></h1>
        <p>Check specific experimental claims against the repository.<br />Every verdict comes with the code lines behind it.</p>
      </section>

      <section className="workspace">
        <div className="workspace-head">
          <div>
            <div className="eyebrow">ANALYSIS WORKSPACE</div>
            <h2>Paper-to-code report</h2>
          </div>
          <label className="repo-picker">
            <span>REPOSITORY</span>
            <select value={repoName} onChange={(event) => setRepoName(event.target.value)} disabled={loadingRepos}>
              <option value="">{loadingRepos ? "Loading repositories…" : "Choose a repository"}</option>
              {repos.map((repo) => (
                <option key={repo.name} value={repo.name} disabled={!repo.has_report}>
                  {repo.repo}{repo.has_report ? "" : " · no report"}
                </option>
              ))}
            </select>
          </label>
        </div>

        {error && <div className="notice error">{error}</div>}
        {!error && !loadingReport && !report && (
          <div className="empty-state">
            <div className="empty-icon">↗</div>
            <h3>{repoName ? "No report for this repository yet" : "Choose a repository to begin"}</h3>
            <p>Run the analysis pipeline to generate a claim-by-claim evidence report.</p>
          </div>
        )}
        {loadingReport && <div className="empty-state"><div className="spinner" />Loading report…</div>}
        {report && (
          <>
            <div className="report-meta">
              <div><span className="meta-label">SOURCE</span><strong>{report.repo}</strong><span className="meta-divider">/</span><span>{report.paper}</span></div>
              <div className="verdict-counts">
                <span className="count supported">{counts.supported} supported</span>
                <span className="count contradicted">{counts.contradicted} contradicted</span>
                <span className="count not_found">{counts.not_found} not found</span>
              </div>
            </div>
            <div className="analysis-grid">
              <aside className="claim-panel">
                <div className="panel-title"><span>EXTRACTED CLAIMS</span><span className="claim-total">{report.claims.length}</span></div>
                <div className="claim-list">
                  {report.claims.map((item, index) => (
                    <button key={`${item.claim}-${index}`} className={`claim-item ${selected === index ? "active" : ""}`} onClick={() => setSelected(index)}>
                      <span className={`status-dot ${item.verdict}`} />
                      <span className="claim-copy">
                        <span className="claim-type">{item.claim_type}</span>
                        <span className="claim-text">{item.claim}</span>
                        <span className={`verdict-label ${item.verdict}`}>{labels[item.verdict]}</span>
                      </span>
                      <span className="chevron">›</span>
                    </button>
                  ))}
                </div>
              </aside>

              <section className="evidence-panel">
                {claim ? (
                  <>
                    <div className="evidence-heading">
                      <div className="eyebrow">CLAIM {String(selected + 1).padStart(2, "0")} / {String(report.claims.length).padStart(2, "0")}</div>
                      <h3>{claim.claim}</h3>
                      <div className={`verdict-banner ${claim.verdict}`}>
                        <span className={`status-dot ${claim.verdict}`} />
                        <strong>{labels[claim.verdict]}</strong>
                        <span>{claim.claim_type}</span>
                      </div>
                    </div>
                    <div className="reasoning">
                      <div className="section-label">WHY THIS VERDICT</div>
                      <p>{claim.reasoning}</p>
                    </div>
                    <div className="code-section">
                      <div className="section-label">
                        <span>CODE EVIDENCE</span>
                        <span>{claim.evidence.length} {claim.evidence.length === 1 ? "SOURCE" : "SOURCES"}</span>
                      </div>
                      {claim.evidence.length ? claim.evidence.map((evidence, index) => (
                        <article className="code-card" key={`${evidence.file}-${evidence.start_line}-${index}`}>
                          <div className="code-title">
                            <span className="file-icon">⌘</span>
                            <span className="file-path">{evidence.file}</span>
                            <span className="line-range">L{evidence.start_line}–{evidence.end_line}</span>
                          </div>
                          <pre className="code-body"><span className="line-numbers">{Array.from({ length: evidence.end_line - evidence.start_line + 1 }, (_, i) => evidence.start_line + i).join("\n")}</span><code>{evidence.text}</code></pre>
                        </article>
                      )) : <div className="no-evidence">No repository lines could be cited for this claim.</div>}
                    </div>
                  </>
                ) : <div className="empty-state">This report contains no claims.</div>}
              </section>
            </div>
          </>
        )}
      </section>
      <footer><span>REPROLENS</span><span>Checks claims against code · does not rerun experiments</span></footer>
    </main>
  );
}
