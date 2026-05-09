/**
 * Data Analysis — Cygnus pipeline: upload CSV, run via API, view report inline (no iframe).
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
  const [selectedName, setSelectedName] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [reportHtml, setReportHtml] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [validationHint, setValidationHint] = useState<string>('');

  const handleFileChange = useCallback(() => {
    const input = fileInputRef.current;
    const next = input?.files?.[0] ?? null;
    setFile(next);
    setSelectedName(next ? next.name : '');
    setValidationHint('');
    setErrorMessage('');
    if (phase === 'done' || phase === 'error') {
      setPhase('idle');
      setReportHtml('');
    }
  }, [phase]);

  const runPipeline = useCallback(async () => {
    if (!file) {
      setValidationHint('Please select a CSV file first.');
      return;
    }
    setValidationHint('');
    setErrorMessage('');
    setReportHtml('');
    setPhase('running');

    const formData = new FormData();
    formData.append('file', file);

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
  }, [file]);

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
        <h1>Data Analysis — Cygnus Report</h1>
        <div className="metadata" style={{ flexWrap: 'wrap', gap: '0.75rem' }}>
          <div className="input-group">
            <label htmlFor="cygnus-csv-input">CSV table:</label>
            <input
              id="cygnus-csv-input"
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileChange}
              disabled={phase === 'running'}
            />
            {selectedName ? (
              <span style={{ marginLeft: '0.5rem', color: '#2c3e50', fontWeight: 500 }}>
                {selectedName}
              </span>
            ) : null}
          </div>
          <button
            type="button"
            className="step-button active"
            onClick={runPipeline}
            disabled={phase === 'running'}
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
          <p>Running Cygnus pipeline… This may take several minutes. Do not close this tab.</p>
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
