/**
 * Data Analysis: upload CSV / Excel, run via API, view report inline (no iframe).
 */

import React, { useRef, useState, useCallback, useEffect } from 'react';
import { getApiBase } from '../lib/apiBase';
import '../styles/AlignmentViewer.css';

type Phase = 'idle' | 'running' | 'done' | 'error';

/** Inner HTML of <body> for injection into the SPA. */
function extractBodyContent(fullHtml: string): string {
  const bodyMatch = fullHtml.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  if (bodyMatch) {
    return bodyMatch[1];
  }
  return fullHtml;
}

/**
 * Copy <style> nodes from parsed document into container (report head styles).
 */
function injectHeadStyles(container: HTMLElement, doc: Document): void {
  doc.head.querySelectorAll('style').forEach((styleEl) => {
    container.appendChild(styleEl.cloneNode(true));
  });
}

/**
 * Set inner HTML and re-insert <script> nodes so browsers execute them (needed for Plotly).
 * React dangerouslySetInnerHTML does not run scripts.
 */
function injectBodyWithExecutableScripts(wrapper: HTMLElement, bodyInnerHtml: string): void {
  wrapper.innerHTML = bodyInnerHtml;
  const scripts = wrapper.querySelectorAll('script');
  scripts.forEach((oldScript) => {
    const next = document.createElement('script');
    Array.from(oldScript.attributes).forEach((attr) => {
      next.setAttribute(attr.name, attr.value);
    });
    next.textContent = oldScript.textContent;
    oldScript.parentNode?.replaceChild(next, oldScript);
  });
}

/** Move all child nodes from source to target (preserves order). */
function moveAllChildren(source: HTMLElement, target: HTMLElement): void {
  while (source.firstChild) {
    target.appendChild(source.firstChild);
  }
}

const DataAnalysis: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const reportMountRef = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [files, setFiles] = useState<File[]>([]);
  const [reportHtml, setReportHtml] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [validationHint, setValidationHint] = useState<string>('');

  const handleFileChange = useCallback(() => {
    const input = fileInputRef.current;
    const list = input?.files ? Array.from(input.files) : [];
    setFiles(list);
    setValidationHint('');
    setErrorMessage('');
    if (phase === 'done' || phase === 'error') {
      setPhase('idle');
      setReportHtml('');
    }
  }, [phase]);

  const runPipeline = useCallback(async () => {
    const input = fileInputRef.current;
    const toSend = input?.files?.length ? Array.from(input.files) : files;
    if (toSend.length === 0) {
      setValidationHint('Please select at least one CSV or Excel file.');
      return;
    }
    setValidationHint('');
    setErrorMessage('');
    setReportHtml('');
    setPhase('running');

    const formData = new FormData();
    for (const f of toSend) {
      formData.append('files', f);
    }

    try {
      const response = await fetch(`${getApiBase()}/api/cygnus/run`, {
        method: 'POST',
        body: formData,
      });
      let data: { success?: boolean; report_html?: string; message?: string } = {};
      try {
        data = await response.json();
      } catch {
        setErrorMessage(`Server returned non-JSON (HTTP ${response.status}).`);
        setPhase('error');
        return;
      }

      if (!response.ok || !data.success) {
        const msg =
          typeof data.message === 'string' && data.message.length > 0
            ? data.message
            : `Request failed (HTTP ${response.status}).`;
        setErrorMessage(msg);
        setPhase('error');
        return;
      }

      const html = data.report_html;
      if (!html || typeof html !== 'string') {
        setErrorMessage('No report HTML returned from server.');
        setPhase('error');
        return;
      }

      setReportHtml(html);
      setPhase('done');
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErrorMessage(`Network or client error: ${msg}`);
      setPhase('error');
    }
  }, [files]);

  useEffect(() => {
    const root = reportMountRef.current;
    if (!root || phase !== 'done' || !reportHtml) {
      return;
    }

    root.innerHTML = '';

    const parser = new DOMParser();
    const doc = parser.parseFromString(reportHtml, 'text/html');

    const scope = document.createElement('div');
    scope.className = 'cygnus-report';
    root.appendChild(scope);

    injectHeadStyles(scope, doc);

    const bodyStaging = document.createElement('div');
    injectBodyWithExecutableScripts(bodyStaging, extractBodyContent(reportHtml));
    moveAllChildren(bodyStaging, scope);

    return () => {
      root.innerHTML = '';
    };
  }, [phase, reportHtml]);

  return (
    <div className="alignment-viewer data-analysis-tab">
      <header className="viewer-header">
        <h1>Data Analysis</h1>
        <div className="metadata" style={{ flexWrap: 'wrap', gap: '0.75rem' }}>
          <div className="input-group" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.35rem' }}>
            <label htmlFor="data-analysis-file-input">Tables (CSV / Excel, multiple allowed):</label>
            <span style={{ color: '#7f8c8d', fontSize: '0.85rem', maxWidth: '42rem' }}>
              Upload the Results Viewer Excel (Full / three-sheet) or CSV with *_positive and *_score columns. Select
              multiple files in one go: use <strong>Ctrl</strong> (Windows/Linux) or <strong>Cmd</strong> (macOS) while
              clicking in the file dialog, or Shift-click for a range. All selected files are merged on the server when
              you run the pipeline.
            </span>
            <input
              id="data-analysis-file-input"
              ref={fileInputRef}
              name="files"
              type="file"
              accept=".csv,.xlsx,.xls,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              multiple={true}
              onChange={handleFileChange}
              disabled={phase === 'running'}
            />
            {files.length > 0 ? (
              <ul
                style={{
                  margin: '0.25rem 0 0',
                  paddingLeft: '1.25rem',
                  color: '#2c3e50',
                  fontSize: '0.9rem',
                  maxWidth: '42rem',
                }}
              >
                {files.map((f) => (
                  <li key={`${f.name}-${f.size}-${f.lastModified}`}>{f.name}</li>
                ))}
              </ul>
            ) : (
              <span style={{ color: '#7f8c8d', fontSize: '0.9rem' }}>No files selected.</span>
            )}
          </div>
          <button
            type="button"
            className="step-button active"
            onClick={runPipeline}
            disabled={phase === 'running' || files.length === 0}
          >
            Run pipeline
          </button>
        </div>
      </header>

      {validationHint ? (
        <p style={{ color: '#c0392b', marginTop: '0.75rem', fontWeight: 600 }}>{validationHint}</p>
      ) : null}

      {phase === 'running' ? (
        <div className="alignment-viewer loading" style={{ minHeight: 120 }}>
          <div className="spinner" />
          <p>
            Running analysis pipeline
            {(() => {
              const n = fileInputRef.current?.files?.length ?? files.length;
              return n > 1 ? ` (merging ${n} files)` : '';
            })()}
            … This may take several minutes. Do not close
            this tab.
          </p>
        </div>
      ) : null}

      {phase === 'error' && errorMessage ? (
        <div
          className="step-content"
          style={{
            marginTop: '1rem',
            padding: '1rem',
            backgroundColor: '#fdeded',
            border: '1px solid #e74c3c',
            borderRadius: 6,
            color: '#c0392b',
            whiteSpace: 'pre-wrap',
            fontFamily: 'monospace',
            fontSize: '0.85rem',
          }}
        >
          {errorMessage}
        </div>
      ) : null}

      {phase === 'done' && reportHtml ? (
        <div
          ref={reportMountRef}
          className="cygnus-report-mount"
          style={{
            marginTop: 24,
            backgroundColor: '#fff',
            borderRadius: 8,
            boxShadow: '0 1px 6px rgba(0,0,0,0.08)',
            overflow: 'hidden',
          }}
        />
      ) : null}
    </div>
  );
};

export default DataAnalysis;
