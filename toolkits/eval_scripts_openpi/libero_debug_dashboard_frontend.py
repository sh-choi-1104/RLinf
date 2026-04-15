#!/usr/bin/env python3

def _render_select_options(options: list[str], selected: str) -> str:
    html_parts = []
    for option in options:
        selected_attr = ' selected="selected"' if option == selected else ""
        html_parts.append(f'<option value="{option}"{selected_attr}>{option}</option>')
    return "".join(html_parts)


def render_dashboard_html(refresh_ms: int, build_token: str, available_libero_types: list[str], available_suites: list[str]) -> str:
    default_type_options = _render_select_options(available_libero_types, "plus")
    default_suite_options = _render_select_options(available_suites, "libero_goal")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LIBERO Debug Control Panel</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #161b22;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --border: #30363d;
      --ok: #3fb950;
      --warn: #d29922;
    }}
    body {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: var(--bg);
      color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(340px, 0.88fr) minmax(840px, 1.52fr);
      gap: 16px;
      padding: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    h1, h2 {{
      margin: 0 0 10px 0;
      font-size: 16px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    .frontend-error {{
      margin-top: 8px;
      color: #ffb4b4;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .links a {{
      color: var(--accent);
      font-size: 12px;
      text-decoration: none;
    }}
    .links a:hover {{
      text-decoration: underline;
    }}
    .progress-block {{
      margin-top: 10px;
    }}
    .catalog-block {{
      margin-top: 10px;
      display: grid;
      gap: 8px;
    }}
    .catalog-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      background: #0b1220;
    }}
    .catalog-title {{
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }}
    .catalog-value {{
      font-size: 12px;
      line-height: 1.45;
      color: var(--text);
      white-space: pre-wrap;
    }}
    .progress-label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .progress-track {{
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: #0b1220;
      border: 1px solid var(--border);
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #3fb950, #58a6ff);
      transition: width 180ms ease;
    }}
    .preview-controls {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 10px 0;
    }}
    .media-stage {{
      display: flex;
      justify-content: center;
      align-items: center;
      width: 100%;
      margin-bottom: 8px;
    }}
    .live-layout {{
      display: grid;
      grid-template-columns: minmax(250px, 320px) minmax(180px, 220px) minmax(260px, 1fr);
      grid-template-areas:
        "media observation status"
        "editor editor editor";
      gap: 14px;
      align-items: start;
    }}
    .live-video-wrap {{
      grid-area: media;
      display: grid;
      gap: 8px;
      justify-items: center;
    }}
    .observation-wrap {{
      grid-area: observation;
      display: grid;
      gap: 10px;
      align-content: start;
    }}
    .obs-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
      background: #0b1220;
      display: grid;
      gap: 6px;
    }}
    .obs-title {{
      color: var(--muted);
      font-size: 11px;
    }}
    .obs-image {{
      width: 100%;
      max-width: 220px;
      max-height: 150px;
      object-fit: contain;
      justify-self: center;
      background: #05080d;
    }}
    .obs-path {{
      font-size: 11px;
      word-break: break-word;
    }}
    .status-wrap {{
      grid-area: status;
      display: grid;
      gap: 8px;
      align-content: start;
    }}
    .path {{
      word-break: break-all;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .status-ok {{
      color: var(--ok);
    }}
    .status-warn {{
      color: var(--warn);
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    label {{
      display: grid;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    input, select, textarea {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #0b1220;
      color: var(--text);
      padding: 8px;
      font: inherit;
    }}
    textarea {{
      min-height: 64px;
      resize: vertical;
    }}
    .checks {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .checks label {{
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--text);
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    button {{
      background: transparent;
      color: var(--accent);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      text-align: left;
      cursor: pointer;
      font: inherit;
    }}
    button:hover {{
      border-color: var(--accent);
    }}
    button.active {{
      border-color: var(--accent);
      color: var(--text);
      background: rgba(88, 166, 255, 0.12);
    }}
    button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    img, video {{
      width: 100%;
      min-width: 0;
      max-width: 420px;
      max-height: 70vh;
      object-fit: contain;
      background: #000;
      border-radius: 8px;
    }}
    .live-frame {{
      max-height: 34vh;
    }}
    .live-video {{
      max-width: 320px;
      max-height: 34vh;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
    }}
    .status-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 8px;
      background: #0b1220;
    }}
    .status-card .label {{
      color: var(--muted);
      font-size: 10px;
      margin-bottom: 3px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .status-card .value {{
      font-size: 12px;
      color: var(--text);
      word-break: break-word;
    }}
    .status-details {{
      display: grid;
      gap: 6px;
      margin-top: 4px;
    }}
    .status-detail {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #0b1220;
      padding: 6px 8px;
    }}
    .status-detail .label {{
      color: var(--muted);
      font-size: 10px;
      margin-bottom: 2px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .status-detail .value {{
      font-size: 12px;
      line-height: 1.35;
      word-break: break-word;
    }}
    .obs-state-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-top: 4px;
    }}
    .obs-state-cell {{
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 7px;
      background: rgba(255, 255, 255, 0.02);
      padding: 5px 6px;
    }}
    .obs-state-label {{
      color: var(--muted);
      font-size: 10px;
      margin-bottom: 2px;
      word-break: break-word;
    }}
    .obs-state-value {{
      font-size: 12px;
      color: var(--text);
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      margin: 0;
      max-height: 46vh;
      overflow: auto;
    }}
    .list {{
      display: grid;
      gap: 8px;
      max-height: 38vh;
      overflow: auto;
    }}
    .action-log {{
      grid-area: editor;
      margin-top: 10px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #0b1220;
      max-height: 38vh;
      overflow-x: auto;
      overflow-y: auto;
    }}
    .action-table {{
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      font-size: 12px;
      table-layout: fixed;
    }}
    .action-table th,
    .action-table td {{
      border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      padding: 4px 6px;
      text-align: right;
      white-space: nowrap;
    }}
    .action-table th:first-child,
    .action-table td:first-child {{
      text-align: left;
    }}
    .action-table tr.current-row td {{
      background: rgba(88, 166, 255, 0.12);
      color: var(--text);
    }}
    .action-table tr.executed-row td:first-child {{
      color: var(--ok);
    }}
    .action-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .action-toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .editor-hint {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 2px;
    }}
    .action-input {{
      width: 100%;
      min-width: 72px;
      padding: 4px 5px;
      text-align: right;
      border-radius: 6px;
      font-size: 11px;
      appearance: textfield;
      -moz-appearance: textfield;
    }}
    .action-input::-webkit-outer-spin-button,
    .action-input::-webkit-inner-spin-button {{
      -webkit-appearance: none;
      margin: 0;
    }}
    .action-input:disabled {{
      opacity: 0.65;
    }}
    details.raw-status {{
      margin-top: 10px;
    }}
    details.raw-status summary {{
      cursor: pointer;
      color: var(--accent);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    @media (max-width: 900px) {{
      img, video {{
        width: 100%;
        min-width: 0;
        max-width: none;
      }}
      .live-layout {{
        grid-template-columns: 1fr;
        grid-template-areas:
          "media"
          "observation"
          "status"
          "editor";
      }}
      .status-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .obs-state-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel">
      <h1>Run Controls</h1>
      <div class="muted" id="sessionInfo"></div>
      <div class="muted" id="buildInfo">Build: {build_token}</div>
      <div class="path" id="runDir"></div>
      <div class="frontend-error" id="frontendError"></div>
      <form id="launchForm">
        <div class="form-grid">
          <label>Libero Type
            <select id="liberoType">{default_type_options}</select>
          </label>
          <label>Suite
            <select id="suiteName">{default_suite_options}</select>
          </label>
          <label>Task ID
            <input id="taskId" type="number" min="0" step="1" value="" placeholder="choose task id" />
          </label>
          <label>Trial Index
            <input id="trialIdx" type="number" min="0" step="1" value="0" />
          </label>
        </div>
        <label class="full">Model Ckpt
          <input
            id="modelPath"
            type="text"
            value="/data/models/pi05_libero_finetuned_v044"
            placeholder="/data/models/pi05_libero_finetuned_v044"
          />
        </label>
        <div class="actions">
          <button type="button" id="setTaskBtn">Set Task</button>
          <button type="button" id="resetRunBtn">Reset</button>
        </div>
        <div class="catalog-block">
          <div class="catalog-card">
            <div class="catalog-title">Selected Task Info</div>
            <div class="catalog-value" id="selectedTaskInfo">Loading task metadata...</div>
          </div>
          <div class="catalog-card">
            <div class="catalog-title">Suite Task Counts</div>
            <div class="catalog-value" id="suiteCountsInfo">Loading suite counts...</div>
          </div>
        </div>
      </form>
      <div class="muted" id="launchStatus"></div>
      <div class="progress-block">
        <div class="progress-label" id="launchProgressLabel">Idle</div>
        <div class="progress-track">
          <div class="progress-fill" id="launchProgressFill"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Live Chunk / Status</h2>
      <div class="live-layout">
        <div class="live-video-wrap">
          <div class="media-stage">
            <img id="liveGif" class="live-video" style="display:none;" />
            <video id="liveVideo" class="live-video" controls autoplay muted loop playsinline style="display:none;"></video>
          </div>
          <div class="muted" id="liveVideoPath"></div>
          <div class="links">
            <a id="liveVideoLink" target="_blank" rel="noopener noreferrer">Open Live MP4</a>
            <a id="liveGifLink" target="_blank" rel="noopener noreferrer">Open Live GIF</a>
          </div>
          <div class="action-toolbar">
            <button type="button" id="simulateBtn">Simulate Chunk</button>
            <button type="button" id="runChunkBtn">Run Chunk</button>
            <button type="button" id="resetEditsBtn">Reset Edits</button>
          </div>
        </div>
        <div class="observation-wrap">
          <div class="obs-card">
            <div class="obs-title">Agentview Observation</div>
            <img id="observationImage" class="obs-image" style="display:none;" />
            <div class="muted obs-path" id="observationImagePath">No agentview observation yet</div>
            <div class="links">
              <a id="observationImageLink" target="_blank" rel="noopener noreferrer">Open Agentview</a>
            </div>
          </div>
          <div class="obs-card">
            <div class="obs-title">Wrist Observation</div>
            <img id="wristObservationImage" class="obs-image" style="display:none;" />
            <div class="muted obs-path" id="wristObservationImagePath">No wrist observation yet</div>
            <div class="links">
              <a id="wristObservationImageLink" target="_blank" rel="noopener noreferrer">Open Wrist</a>
            </div>
          </div>
        </div>
        <div class="status-wrap">
          <div class="status-grid" id="statusGrid"></div>
          <div class="status-details" id="statusDetails"></div>
        </div>
        <div class="action-log">
          <div class="action-meta" id="actionMeta">No action plan yet</div>
          <div class="editor-hint" id="actionEditorHint">A fresh chunk will become editable here when the policy is ready.</div>
          <div id="actionTableWrap">No action plan yet</div>
        </div>
      </div>
      <details class="raw-status">
        <summary>Raw Status</summary>
        <pre id="liveStatus"></pre>
      </details>
    </section>

    <section class="panel">
      <h1>Completed Chunk Preview</h1>
      <div class="muted" id="selectedChunkLabel">No completed chunk yet</div>
      <div class="preview-controls">
        <button type="button" id="prevChunkBtn">Prev Saved</button>
        <button type="button" id="nextChunkBtn">Next Saved</button>
        <button type="button" id="latestChunkBtn">Latest Chunk</button>
        <button type="button" id="playSequenceBtn">Play 0 → Latest</button>
        <button type="button" id="stopSequenceBtn">Stop Sequence</button>
      </div>
      <div class="media-stage">
        <img id="gif" style="display:none;" />
        <video id="video" controls autoplay muted loop playsinline style="display:none;"></video>
      </div>
      <div class="muted" id="videoPath"></div>
      <div class="links">
        <a id="gifLink" target="_blank" rel="noopener noreferrer">Open GIF</a>
        <a id="mp4Link" target="_blank" rel="noopener noreferrer">Open MP4</a>
      </div>
    </section>

    <section class="panel">
      <h2>Chunk Media</h2>
      <div class="list" id="chunkList"></div>
    </section>

    <section class="panel">
      <h2>Worker Log</h2>
      <pre id="tmuxTail"></pre>
    </section>

    <section class="panel">
      <h2>Summary / Trace</h2>
      <pre id="summary"></pre>
      <hr style="border-color:var(--border);margin:12px 0;">
      <pre id="trace"></pre>
    </section>
  </div>
  <script>
    (function() {{
      function setText(id, text) {{
        var el = document.getElementById(id);
        if (el) el.textContent = text;
      }}
      function setHtml(id, html) {{
        var el = document.getElementById(id);
        if (el) el.innerHTML = html;
      }}
      function getJson(url, cb) {{
        var xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.onreadystatechange = function() {{
          if (xhr.readyState !== 4) return;
          if (xhr.status < 200 || xhr.status >= 300) {{
            cb(new Error('HTTP ' + xhr.status));
            return;
          }}
          try {{
            cb(null, JSON.parse(xhr.responseText));
          }} catch (err) {{
            cb(err);
          }}
        }};
        xhr.send();
      }}
      function updateBasic(state) {{
        var control = state.control || {{}};
        var progress = state.progress || {{ percent: 0, label: 'Idle' }};
        var idleSession = !control.session_alive && !control.current_run_dir;
        setHtml(
          'sessionInfo',
          "Session: <span class='" + (control.session_alive ? "status-ok" : "status-warn") + "'>" +
          (control.session_name || "n/a") + "</span> | " +
          (control.session_alive ? "alive" : (idleSession ? "idle - choose a task and press Set Task" : "not running"))
        );
        setText('buildInfo', 'Build: ' + (control.build_token || '{build_token}'));
        setText('runDir', state.run_dir || control.current_run_dir || 'No active run directory yet');
        setText('launchProgressLabel', progress.label + ' (' + progress.percent + '%)');
        var fill = document.getElementById('launchProgressFill');
        if (fill) fill.style.width = String(progress.percent) + '%';
      }}
      function poll() {{
        getJson('/api/state?ts=' + Date.now(), function(err, state) {{
          if (err) {{
            setText('frontendError', 'Frontend polling error: ' + err.message);
            return;
          }}
          updateBasic(state);
        }});
      }}
      window.onerror = function(message, source, lineno, colno) {{
        setText('frontendError', 'Frontend JS error: ' + message + ' @ ' + source + ':' + lineno + ':' + colno);
      }};
      poll();
      setInterval(poll, {refresh_ms});
    }})();
  </script>
  <script>
    const refreshMs = {refresh_ms};
    const actionDimLabels = ["eef_x", "eef_y", "eef_z", "rot_x", "rot_y", "rot_z", "gripper"];
    const observationStateLabels = [
      "eef_pos_x",
      "eef_pos_y",
      "eef_pos_z",
      "eef_axisangle_x",
      "eef_axisangle_y",
      "eef_axisangle_z",
      "gripper_left_qpos",
      "gripper_right_qpos",
    ];
    let controlsInitialized = false;
    let currentChunkItems = [];
    let followLatestChunk = true;
    let selectedChunkIndex = null;
    let sequencePlaying = false;
    let sequenceQueue = [];
    let sequenceQueueIndex = -1;
    let previewVideoVersion = null;
    let previewGifVersion = null;
    let liveVideoVersion = null;
    let liveVideoPathValue = null;
    let liveGifVersion = null;
    let liveGifPathValue = null;
    let observationImageVersion = null;
    let observationImagePathValue = null;
    let wristObservationImageVersion = null;
    let wristObservationImagePathValue = null;
    let editableActions = null;
    let editableSourceActions = null;
    let editableChunkKey = null;
    let lastRenderedPlan = null;
    let lastActionEditorRenderKey = null;
    let lastState = null;
    let taskCatalogCache = {{}};
    let taskCatalogPromiseCache = {{}};

    function artifactUrl(relPath) {{
      if (!relPath) return "";
      return "/artifacts/" + relPath.split("/").map(encodeURIComponent).join("/") + "?ts=" + Date.now();
    }}

    function prettyValue(value) {{
      if (value === undefined || value === null) return "-";
      if (typeof value === "number") return value.toFixed(4);
      if (Array.isArray(value)) return value.map((item) => prettyValue(item)).join(", ");
      return String(value);
    }}

    function actionLabelsForWidth(numDims, supplied) {{
      const base = (supplied && supplied.length) ? supplied : actionDimLabels;
      const labels = [];
      for (let idx = 0; idx < numDims; idx += 1) {{
        labels.push(base[idx] || ("a" + idx));
      }}
      return labels;
    }}

    function deepCopyActions(actions) {{
      if (!actions) return null;
      return actions.map((row) => row.map((value) => Number(value)));
    }}

    async function getTaskCatalog(liberoType) {{
      if (taskCatalogCache[liberoType]) {{
        return taskCatalogCache[liberoType];
      }}
      if (taskCatalogPromiseCache[liberoType]) {{
        return taskCatalogPromiseCache[liberoType];
      }}
      taskCatalogPromiseCache[liberoType] = fetch(
        "/api/task_catalog?libero_type=" + encodeURIComponent(liberoType)
      ).then(async function(resp) {{
        const data = await resp.json();
        if (!resp.ok) {{
          throw new Error(data.error || ("task catalog failed: " + resp.status));
        }}
        taskCatalogCache[liberoType] = data;
        delete taskCatalogPromiseCache[liberoType];
        return data;
      }}).catch(function(err) {{
        delete taskCatalogPromiseCache[liberoType];
        throw err;
      }});
      return taskCatalogPromiseCache[liberoType];
    }}

    function phaseAllowsSimulation(phase) {{
      return phase === "awaiting_chunk_execution";
    }}

    function phaseAllowsRun(phase) {{
      return phase === "awaiting_chunk_execution";
    }}

    function phaseAllowsSetTask(phase) {{
      return (
        phase === "awaiting_chunk_execution" ||
        phase === "chunk_complete" ||
        phase === "finished"
      );
    }}

    function extractPlan(state) {{
      const liveStatus = state.live_status || {{}};
      const traceTail = state.trace_tail || [];
      let planActions = liveStatus.current_plan_actions || liveStatus.executed_actions || null;
      let predictedActions = liveStatus.predicted_actions || null;
      let labels = liveStatus.action_dim_labels || actionDimLabels;
      let chunkIdx = liveStatus.chunk_idx;
      let planSource = "live status";

      if (!planActions) {{
        for (let idx = traceTail.length - 1; idx >= 0; idx -= 1) {{
          const record = traceTail[idx];
          if (record.event === "plan") {{
            planActions = record.executed_actions || record.predicted_actions || null;
            predictedActions = record.predicted_actions || null;
            chunkIdx = record.chunk_idx;
            planSource = "trace";
            break;
          }}
        }}
      }}

      if (!planActions || !planActions.length) {{
        return null;
      }}

      return {{
        actions: planActions,
        originalActions: predictedActions || planActions,
        predictedActions: predictedActions,
        labels: actionLabelsForWidth(planActions[0].length, labels),
        chunkIdx: chunkIdx,
        planSource: planSource,
        lastStep: liveStatus.chunk_step,
        editHistory: liveStatus.edit_history || [],
        phase: String(liveStatus.phase || ""),
      }};
    }}

    function buildPlanKey(state, plan) {{
      const liveStatus = state.live_status || {{}};
      return [
        String(liveStatus.libero_type || ""),
        String(liveStatus.suite_name || ""),
        String(liveStatus.task_id || ""),
        String(liveStatus.trial_idx || ""),
        String(plan.chunkIdx === undefined || plan.chunkIdx === null ? "" : plan.chunkIdx),
        JSON.stringify(plan.originalActions),
      ].join("|");
    }}

    function syncEditableActions(state, plan) {{
      if (!plan) {{
        editableActions = null;
        editableSourceActions = null;
        editableChunkKey = null;
        lastRenderedPlan = null;
        lastActionEditorRenderKey = null;
        return;
      }}

      const nextKey = buildPlanKey(state, plan);
      if (nextKey !== editableChunkKey) {{
        editableChunkKey = nextKey;
        editableSourceActions = deepCopyActions(plan.originalActions);
        editableActions = deepCopyActions(plan.actions);
      }}
      lastRenderedPlan = plan;
    }}

    function buildStatusCards(state) {{
      const liveStatus = state.live_status || {{}};
      const summary = state.summary || {{}};
      const cards = [
        ["Phase", liveStatus.phase || summary.stop_reason || "-"],
        ["Type", liveStatus.libero_type || summary.libero_type || "-"],
        ["Suite", liveStatus.suite_name || summary.suite_name || "-"],
        ["Task ID", liveStatus.task_id === undefined ? (summary.task_id === undefined ? "-" : summary.task_id) : liveStatus.task_id],
        ["Trial", liveStatus.trial_idx === undefined ? (summary.trial_idx === undefined ? "-" : summary.trial_idx) : liveStatus.trial_idx],
        ["Taxonomy", liveStatus.taxonomy || summary.taxonomy || "-"],
        ["Chunk", liveStatus.chunk_idx === undefined ? "-" : liveStatus.chunk_idx],
        ["Step", liveStatus.chunk_step === undefined ? "-" : liveStatus.chunk_step],
        ["Env", liveStatus.env_step === undefined ? "-" : liveStatus.env_step],
        ["Done", liveStatus.done === undefined ? (summary.success === undefined ? "-" : String(summary.success)) : String(liveStatus.done)],
        ["Model", liveStatus.model_reused ? "reused" : "loaded"],
      ];
      return cards
        .map(function(pair) {{
          return '<div class="status-card"><div class="label">' + pair[0] + '</div><div class="value">' + prettyValue(pair[1]) + '</div></div>';
        }})
        .join("");
    }}

    function buildStatusDetails(state) {{
      const liveStatus = state.live_status || {{}};
      const summary = state.summary || {{}};
      const details = [];
      const taskName = liveStatus.task_name || summary.task_name || "";
      const taskDescription = liveStatus.task_description || summary.task_description || "";
      const observationState = liveStatus.observation_state || null;
      const observationLabels = liveStatus.observation_state_labels || observationStateLabels;
      const simulationError = liveStatus.simulation_error || "";

      if (observationState && observationState.length) {{
        const obsGrid = observationState
          .map(function(value, idx) {{
            const label = observationLabels[idx] || ("obs_" + idx);
            return (
              '<div class="obs-state-cell">' +
              '<div class="obs-state-label">' + label + '</div>' +
              '<div class="obs-state-value">' + Number(value).toFixed(6) + "</div>" +
              "</div>"
            );
          }})
          .join("");
        details.push([
          "Observation State (8D)",
          '<div class="obs-state-grid">' + obsGrid + "</div>",
        ]);
      }}

      if (taskName) {{
        details.push(["Task", taskName]);
      }}
      if (taskDescription) {{
        details.push(["Prompt", taskDescription]);
      }}
      if (simulationError) {{
        details.push(["Sim Error", simulationError]);
      }}

      if (!details.length) {{
        return '<div class="status-detail"><div class="label">Status</div><div class="value">No detailed live status yet</div></div>';
      }}

      return details
        .map(function(pair) {{
          return '<div class="status-detail"><div class="label">' + pair[0] + '</div><div class="value">' + pair[1] + '</div></div>';
        }})
        .join("");
    }}

    async function updateRunControlMeta() {{
      const selectedTaskInfoEl = document.getElementById("selectedTaskInfo");
      const suiteCountsInfoEl = document.getElementById("suiteCountsInfo");
      const liberoType = document.getElementById("liberoType").value;
      const suiteName = document.getElementById("suiteName").value;
      const rawTaskId = document.getElementById("taskId").value;
      const trimmedTaskId = String(rawTaskId || "").trim();
      const taskId = trimmedTaskId === "" ? null : Number(trimmedTaskId);

      selectedTaskInfoEl.textContent = "Loading task metadata...";
      suiteCountsInfoEl.textContent = "Loading suite counts...";

      try {{
        const catalog = await getTaskCatalog(liberoType);
        const suiteCounts = catalog.suite_task_counts || {{}};
        const suiteTaxonomies = catalog.task_taxonomy_by_suite || {{}};
        const countLines = [];
        for (let idx = 0; idx < (lastState && lastState.control && lastState.control.available_suites ? lastState.control.available_suites.length : 0); idx += 1) {{
          const suite = lastState.control.available_suites[idx];
          const count = suiteCounts[suite];
          if (count === undefined || count === null) {{
            countLines.push(suite + ": unavailable");
          }} else {{
            countLines.push(suite + ": " + count + " tasks");
          }}
        }}
        if (!countLines.length) {{
          for (const suite in suiteCounts) {{
            const count = suiteCounts[suite];
            countLines.push(suite + ": " + count + " tasks");
          }}
        }}
        suiteCountsInfoEl.textContent = countLines.join("\\n");

        const suiteOptions = document.getElementById("suiteName").options;
        for (let idx = 0; idx < suiteOptions.length; idx += 1) {{
          const option = suiteOptions[idx];
          const baseValue = option.value;
          const count = suiteCounts[baseValue];
          option.textContent = (count === undefined || count === null)
            ? baseValue
            : (baseValue + " (" + count + ")");
        }}

        const suiteCount = suiteCounts[suiteName];
        if (trimmedTaskId === "") {{
          selectedTaskInfoEl.textContent = "Task ID is empty. Choose a task to inspect metadata.";
          return;
        }}
        if (!Number.isFinite(taskId)) {{
          selectedTaskInfoEl.textContent = "Enter a valid integer Task ID.";
          return;
        }}
        if (suiteCount === undefined || suiteCount === null) {{
          selectedTaskInfoEl.textContent = "Task count is unavailable for " + suiteName + ".";
          return;
        }}
        if (taskId < 0 || taskId >= suiteCount) {{
          selectedTaskInfoEl.textContent =
            "Task ID " + taskId + " is out of range for " + suiteName + ". Valid range: 0.." + String(suiteCount - 1);
          return;
        }}

        const infoLines = [
          "Suite: " + suiteName,
          "Task Count: " + suiteCount,
          "Task ID: " + taskId,
        ];
        if (liberoType === "plus") {{
          const taxonomy = (suiteTaxonomies[suiteName] || {{}})[String(taskId)] || "Unknown";
          infoLines.push("Taxonomy: " + taxonomy);
        }} else {{
          infoLines.push("Taxonomy: n/a for " + liberoType);
        }}
        selectedTaskInfoEl.textContent = infoLines.join("\\n");
      }} catch (err) {{
        selectedTaskInfoEl.textContent = "Task metadata error: " + err.message;
        suiteCountsInfoEl.textContent = "Suite counts error: " + err.message;
      }}
    }}

    function renderActionEditor(state) {{
      const liveStatus = state.live_status || {{}};
      const phase = String(liveStatus.phase || "");
      const metaEl = document.getElementById("actionMeta");
      const hintEl = document.getElementById("actionEditorHint");
      const wrapEl = document.getElementById("actionTableWrap");
      const plan = extractPlan(state);

      syncEditableActions(state, plan);
      updateControlButtons(phase, !!plan);

      if (!plan || !editableActions) {{
        metaEl.textContent = "No action plan available yet";
        hintEl.textContent = "A fresh chunk will become editable here when the policy is ready.";
        wrapEl.innerHTML = "No action plan available yet";
        lastActionEditorRenderKey = null;
        return;
      }}

      const editable = phaseAllowsSimulation(phase);
      const renderKey = editableChunkKey + "|editable=" + String(editable);
      const header = ['<tr><th>Step</th>']
        .concat(plan.labels.map(function(label) {{ return "<th>" + label + "</th>"; }}))
        .concat(["</tr>"])
        .join("");

      const rows = [];
      for (let stepIdx = 0; stepIdx < editableActions.length; stepIdx += 1) {{
        const action = editableActions[stepIdx];
        const classes = [];
        if (typeof plan.lastStep === "number" && stepIdx === plan.lastStep) {{
          classes.push("current-row");
        }}
        if (typeof plan.lastStep === "number" && stepIdx <= plan.lastStep) {{
          classes.push("executed-row");
        }}
        const cells = ['<td>[' + String(stepIdx).padStart(2, "0") + ']</td>'];
        for (let dimIdx = 0; dimIdx < action.length; dimIdx += 1) {{
          const value = Number(action[dimIdx]);
          const disabledAttr = editable ? "" : ' disabled="disabled"';
          cells.push(
            '<td><input class="action-input" type="text" inputmode="decimal" spellcheck="false" data-step="' +
            stepIdx +
            '" data-dim="' +
            dimIdx +
            '" value="' +
            value.toFixed(6) +
            '"' +
            disabledAttr +
            " /></td>"
          );
        }}
        rows.push(
          "<tr" + (classes.length ? ' class=\"' + classes.join(" ") + '\"' : "") + ">" + cells.join("") + "</tr>"
        );
      }}

      const metaParts = [];
      metaParts.push(
        "chunk=" + String(plan.chunkIdx === undefined || plan.chunkIdx === null ? "?" : plan.chunkIdx).padStart(3, "0")
      );
      metaParts.push("source=" + plan.planSource);
      if (typeof plan.lastStep === "number") {{
        metaParts.push("current_step=" + String(plan.lastStep).padStart(2, "0"));
      }}
      if (plan.editHistory.length) {{
        metaParts.push("backend_edits=" + plan.editHistory.join(" | "));
      }}
      if (plan.predictedActions && plan.predictedActions.length) {{
        metaParts.push(
          "pred0=" + plan.predictedActions[0].map(function(value) {{ return Number(value).toFixed(6); }}).join(", ")
        );
      }}
      metaEl.textContent = metaParts.join(" • ");

      if (editable) {{
        hintEl.textContent = "Edit the chunk values below. Simulate Chunk is a dry-run on the same scene, while Run Chunk advances the real environment and then requests the next policy chunk.";
      }} else if (phase === "simulating_chunk") {{
        hintEl.textContent = "Dry-run simulation is running from the current policy input scene. The real environment is not advancing.";
      }} else if (phase === "planning_next_chunk") {{
        hintEl.textContent = "The real environment advanced. A fresh policy chunk is being generated for the next scene.";
      }} else if (phase === "executing_chunk") {{
        hintEl.textContent = "Run Chunk is executing on the real environment. The live video updates as the robot moves.";
      }} else if (phase === "finished") {{
        hintEl.textContent = "This task finished. Use Set Task to move to another task without reloading the model.";
      }} else {{
        hintEl.textContent = "The editor becomes active when a new chunk is waiting for simulation.";
      }}

      const activeEl = document.activeElement;
      const isEditingActionInput =
        !!activeEl &&
        activeEl.classList &&
        activeEl.classList.contains("action-input") &&
        wrapEl.contains(activeEl);
      if (renderKey === lastActionEditorRenderKey && wrapEl.querySelector(".action-table")) {{
        if (isEditingActionInput) {{
          return;
        }}
        return;
      }}

      wrapEl.innerHTML = '<table class="action-table"><thead>' + header + "</thead><tbody>" + rows.join("") + "</tbody></table>";
      lastActionEditorRenderKey = renderKey;

      const inputEls = wrapEl.querySelectorAll(".action-input");
      for (let idx = 0; idx < inputEls.length; idx += 1) {{
        inputEls[idx].addEventListener("input", function(event) {{
          const input = event.target;
          const stepIdx = Number(input.getAttribute("data-step"));
          const dimIdx = Number(input.getAttribute("data-dim"));
          const numericValue = Number(input.value);
          if (Number.isFinite(numericValue)) {{
            editableActions[stepIdx][dimIdx] = numericValue;
          }}
        }});
        inputEls[idx].addEventListener("change", function(event) {{
          const input = event.target;
          const stepIdx = Number(input.getAttribute("data-step"));
          const dimIdx = Number(input.getAttribute("data-dim"));
          const numericValue = Number(input.value);
          if (!Number.isFinite(numericValue)) {{
            input.value = editableActions[stepIdx][dimIdx].toFixed(6);
            return;
          }}
          editableActions[stepIdx][dimIdx] = numericValue;
          input.value = numericValue.toFixed(6);
        }});
      }}
    }}

    function updateControlButtons(phase, hasPlan) {{
      const simulateBtn = document.getElementById("simulateBtn");
      const runChunkBtn = document.getElementById("runChunkBtn");
      const resetBtn = document.getElementById("resetEditsBtn");
      const setTaskBtn = document.getElementById("setTaskBtn");
      const resetRunBtn = document.getElementById("resetRunBtn");
      const sessionAlive = !!(lastState && lastState.control && lastState.control.session_alive);

      simulateBtn.disabled = !sessionAlive || !hasPlan || !phaseAllowsSimulation(phase);
      runChunkBtn.disabled = !sessionAlive || !hasPlan || !phaseAllowsRun(phase);
      resetBtn.disabled = !sessionAlive || !hasPlan || !phaseAllowsSimulation(phase);
      setTaskBtn.disabled = sessionAlive && !phaseAllowsSetTask(phase);
      resetRunBtn.disabled = false;
    }}

    function readEditableActions() {{
      if (!editableActions) return null;
      const nextActions = deepCopyActions(editableActions);
      const inputEls = document.querySelectorAll("#actionTableWrap .action-input");
      for (let idx = 0; idx < inputEls.length; idx += 1) {{
        const input = inputEls[idx];
        const stepIdx = Number(input.getAttribute("data-step"));
        const dimIdx = Number(input.getAttribute("data-dim"));
        const numericValue = Number(input.value);
        if (!Number.isFinite(numericValue)) {{
          throw new Error("One or more action cells contain invalid numbers.");
        }}
        nextActions[stepIdx][dimIdx] = numericValue;
      }}
      editableActions = deepCopyActions(nextActions);
      return nextActions;
    }}

    function parseChunkNumber(relPath, fallbackIndex) {{
      const match = /chunk(\\d+)/i.exec(relPath || "");
      if (match) return Number(match[1]);
      return fallbackIndex;
    }}

    function buildChunkItems(state) {{
      const liveStatus = state.live_status || {{}};
      const summary = state.summary || {{}};
      const suiteName = liveStatus.suite_name || summary.suite_name || "";
      const taskId = liveStatus.task_id === undefined ? summary.task_id : liveStatus.task_id;
      const taskPrefix = suiteName && taskId !== undefined && taskId !== null
        ? suiteName + "_task" + String(taskId).padStart(4, "0") + "_"
        : "";
      const gifList = (state.chunk_gifs || []).filter(function(path) {{
        return !taskPrefix || path.indexOf(taskPrefix) !== -1;
      }});
      const videoList = (state.chunk_videos || []).filter(function(path) {{
        return !taskPrefix || path.indexOf(taskPrefix) !== -1;
      }});
      const total = Math.max(gifList.length, videoList.length);
      const items = [];
      for (let idx = 0; idx < total; idx += 1) {{
        const gif = gifList[idx] || null;
        const video = videoList[idx] || null;
        const anyPath = gif || video || "";
        items.push({{
          index: idx,
          chunkNumber: parseChunkNumber(anyPath, idx),
          gif: gif,
          video: video,
          label: "Chunk " + String(parseChunkNumber(anyPath, idx)).padStart(3, "0"),
        }});
      }}
      return items;
    }}

    function updateChunkLinks(item) {{
      document.getElementById("gifLink").href = item && item.gif ? artifactUrl(item.gif) : "#";
      document.getElementById("mp4Link").href = item && item.video ? artifactUrl(item.video) : "#";
    }}

    function renderChunkPreview(item, options) {{
      const autoplayVideo = options && options.autoplayVideo;
      const gifEl = document.getElementById("gif");
      const videoEl = document.getElementById("video");
      const selectedLabel = document.getElementById("selectedChunkLabel");
      const pathLabel = document.getElementById("videoPath");

      if (!item) {{
        selectedLabel.textContent = "No completed chunk yet";
        pathLabel.textContent = "No chunk media yet";
        gifEl.style.display = "none";
        videoEl.style.display = "none";
        previewVideoVersion = null;
        previewGifVersion = null;
        updateChunkLinks(null);
        return;
      }}

      selectedLabel.textContent = sequencePlaying ? (item.label + " • sequence playback") : (item.label + " • selected");
      pathLabel.textContent = item.video || item.gif || "No chunk media yet";
      updateChunkLinks(item);

      if (autoplayVideo && item.video) {{
        previewVideoVersion = item.video;
        videoEl.loop = false;
        videoEl.style.display = "block";
        gifEl.style.display = "none";
        videoEl.src = artifactUrl(item.video);
        videoEl.load();
        videoEl.play().catch(function() {{}});
        return;
      }}

      if (item.gif) {{
        previewGifVersion = item.gif;
        gifEl.src = artifactUrl(item.gif);
        gifEl.style.display = "block";
        videoEl.style.display = "none";
        return;
      }}

      if (item.video) {{
        previewVideoVersion = item.video;
        videoEl.loop = true;
        videoEl.style.display = "block";
        gifEl.style.display = "none";
        videoEl.src = artifactUrl(item.video);
        videoEl.load();
        videoEl.play().catch(function() {{}});
        return;
      }}

      gifEl.style.display = "none";
      videoEl.style.display = "none";
    }}

    function renderChunkList() {{
      const listEl = document.getElementById("chunkList");
      listEl.innerHTML = "";
      for (let idx = currentChunkItems.length - 1; idx >= 0; idx -= 1) {{
        const item = currentChunkItems[idx];
        const button = document.createElement("button");
        button.textContent = item.label + (item.video ? " • mp4" : "") + (item.gif ? " • gif" : "");
        if (item.index === selectedChunkIndex) {{
          button.classList.add("active");
        }}
        button.onclick = function() {{
          followLatestChunk = false;
          sequencePlaying = false;
          selectedChunkIndex = item.index;
          renderChunkPreview(item);
          renderChunkList();
        }};
        listEl.appendChild(button);
      }}
    }}

    function syncSelectedChunk() {{
      if (!currentChunkItems.length) {{
        selectedChunkIndex = null;
        renderChunkPreview(null);
        renderChunkList();
        return;
      }}
      if (followLatestChunk || selectedChunkIndex === null || selectedChunkIndex >= currentChunkItems.length) {{
        selectedChunkIndex = currentChunkItems.length - 1;
      }}
      if (!sequencePlaying) {{
        renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      }}
      renderChunkList();
    }}

    function playSequenceAt(queueIndex) {{
      if (!sequencePlaying || queueIndex < 0 || queueIndex >= sequenceQueue.length) {{
        sequencePlaying = false;
        sequenceQueue = [];
        sequenceQueueIndex = -1;
        renderChunkPreview(selectedChunkIndex !== null ? currentChunkItems[selectedChunkIndex] : null);
        renderChunkList();
        return;
      }}

      sequenceQueueIndex = queueIndex;
      selectedChunkIndex = sequenceQueue[queueIndex];
      renderChunkPreview(currentChunkItems[selectedChunkIndex], {{ autoplayVideo: true }});
      renderChunkList();
    }}

    function startSequencePlayback() {{
      const playable = currentChunkItems.filter(function(item) {{ return !!item.video; }}).map(function(item) {{ return item.index; }});
      if (!playable.length) {{
        document.getElementById("launchStatus").textContent = "No stable chunk MP4s available yet.";
        return;
      }}
      followLatestChunk = false;
      sequencePlaying = true;
      sequenceQueue = playable;
      playSequenceAt(0);
    }}

    function stopSequencePlayback() {{
      sequencePlaying = false;
      sequenceQueue = [];
      sequenceQueueIndex = -1;
      renderChunkPreview(selectedChunkIndex !== null ? currentChunkItems[selectedChunkIndex] : null);
      renderChunkList();
    }}

    function setOptions(selectEl, values) {{
      if (selectEl.options.length > 0) return;
      for (let idx = 0; idx < values.length; idx += 1) {{
        const option = document.createElement("option");
        option.value = values[idx];
        option.textContent = values[idx];
        selectEl.appendChild(option);
      }}
    }}

    function populateControls(control) {{
      setOptions(document.getElementById("liberoType"), control.available_libero_types || []);
      setOptions(document.getElementById("suiteName"), control.available_suites || []);
      const cfg = control.last_launch_config || control.defaults || {{}};
      document.getElementById("liberoType").value = cfg.libero_type || "plus";
      document.getElementById("suiteName").value = cfg.task_suite_name || "libero_goal";
      document.getElementById("taskId").value = "";
      document.getElementById("trialIdx").value = cfg.trial_idx === undefined || cfg.trial_idx === null ? 0 : cfg.trial_idx;
      document.getElementById("modelPath").value = cfg.model_path || control.current_model_path || "/data/models/pi05_libero_finetuned_v044";
      controlsInitialized = true;
    }}

    function updateLiveVideo(state) {{
      const videoEl = document.getElementById("liveVideo");
      const gifEl = document.getElementById("liveGif");
      const pathEl = document.getElementById("liveVideoPath");
      const linkEl = document.getElementById("liveVideoLink");
      const gifLinkEl = document.getElementById("liveGifLink");
      const liveVideo = state.live_chunk_video || null;
      const version = state.live_chunk_video_mtime_ns || null;
      const liveGif = state.live_chunk_gif || null;
      const gifVersion = state.live_chunk_gif_mtime_ns || null;

      pathEl.textContent = liveGif || liveVideo || "No live chunk media yet";
      linkEl.href = liveVideo ? artifactUrl(liveVideo) : "#";
      gifLinkEl.href = liveGif ? artifactUrl(liveGif) : "#";

      // The live pane prioritizes GIF because browsers can show transient artifacts
      // when an actively rewritten MP4 is reloaded every poll during Run Chunk.
      if (liveGif) {{
        gifEl.style.display = "block";
        videoEl.style.display = "none";
        if (liveGif !== liveGifPathValue || gifVersion !== liveGifVersion) {{
          liveGifPathValue = liveGif;
          liveGifVersion = gifVersion;
          gifEl.src = artifactUrl(liveGif);
        }}
        return;
      }}

      gifEl.style.display = "none";
      liveGifVersion = null;
      liveGifPathValue = null;

      if (!liveVideo) {{
        videoEl.style.display = "none";
        liveVideoVersion = null;
        liveVideoPathValue = null;
        return;
      }}

      videoEl.style.display = "block";
      videoEl.loop = true;
      if (liveVideo !== liveVideoPathValue || version !== liveVideoVersion) {{
        liveVideoPathValue = liveVideo;
        liveVideoVersion = version;
        videoEl.src = artifactUrl(liveVideo);
        videoEl.load();
        videoEl.play().catch(function() {{}});
      }}
    }}

    function updateObservationImages(state) {{
      const observationImageEl = document.getElementById("observationImage");
      const observationPathEl = document.getElementById("observationImagePath");
      const observationLinkEl = document.getElementById("observationImageLink");
      const wristImageEl = document.getElementById("wristObservationImage");
      const wristPathEl = document.getElementById("wristObservationImagePath");
      const wristLinkEl = document.getElementById("wristObservationImageLink");

      const observationImage = state.live_observation_image || null;
      const observationVersion = state.live_observation_image_mtime_ns || null;
      const wristObservationImage = state.live_observation_wrist_image || null;
      const wristObservationVersion = state.live_observation_wrist_image_mtime_ns || null;

      observationPathEl.textContent = observationImage || "No agentview observation yet";
      observationLinkEl.href = observationImage ? artifactUrl(observationImage) : "#";
      if (!observationImage) {{
        observationImageEl.style.display = "none";
        observationImageVersion = null;
        observationImagePathValue = null;
      }} else {{
        observationImageEl.style.display = "block";
        if (
          observationImage !== observationImagePathValue ||
          observationVersion !== observationImageVersion
        ) {{
          observationImagePathValue = observationImage;
          observationImageVersion = observationVersion;
          observationImageEl.src = artifactUrl(observationImage);
        }}
      }}

      wristPathEl.textContent = wristObservationImage || "No wrist observation yet";
      wristLinkEl.href = wristObservationImage ? artifactUrl(wristObservationImage) : "#";
      if (!wristObservationImage) {{
        wristImageEl.style.display = "none";
        wristObservationImageVersion = null;
        wristObservationImagePathValue = null;
      }} else {{
        wristImageEl.style.display = "block";
        if (
          wristObservationImage !== wristObservationImagePathValue ||
          wristObservationVersion !== wristObservationImageVersion
        ) {{
          wristObservationImagePathValue = wristObservationImage;
          wristObservationImageVersion = wristObservationVersion;
          wristImageEl.src = artifactUrl(wristObservationImage);
        }}
      }}
    }}

    async function postJson(path, payload) {{
      const resp = await fetch(path, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload || {{}}),
      }});
      const data = await resp.json();
      if (!resp.ok) {{
        throw new Error(data.error || ("request failed: " + resp.status));
      }}
      return data;
    }}

    async function refresh() {{
      const resp = await fetch("/api/state?ts=" + Date.now());
      const state = await resp.json();
      const control = state.control || {{}};
      const phase = String((state.live_status || {{}}).phase || "");
      lastState = state;

      if (!controlsInitialized) {{
        populateControls(control);
      }}

      const idleSession = !control.session_alive && !control.current_run_dir;
      document.getElementById("sessionInfo").innerHTML =
        "Session: <span class='" + (control.session_alive ? "status-ok" : "status-warn") + "'>" +
        (control.session_name || "n/a") + "</span> | " +
        (control.session_alive ? "alive" : (idleSession ? "idle - choose a task and press Set Task" : "not running"));
      document.getElementById("buildInfo").textContent = "Build: " + (control.build_token || "{build_token}");
      const progress = state.progress || {{ percent: 0, label: "Idle" }};
      document.getElementById("launchProgressLabel").textContent = progress.label + " (" + progress.percent + "%)";
      document.getElementById("launchProgressFill").style.width = String(progress.percent) + "%";
      document.getElementById("runDir").textContent = state.run_dir || control.current_run_dir || "No active run directory yet";
      document.getElementById("summary").textContent = JSON.stringify(state.summary, null, 2);
      document.getElementById("trace").textContent = JSON.stringify(state.trace_tail, null, 2);
      document.getElementById("liveStatus").textContent = JSON.stringify(state.live_status, null, 2);
      document.getElementById("statusGrid").innerHTML = buildStatusCards(state);
      document.getElementById("statusDetails").innerHTML = buildStatusDetails(state);
      document.getElementById("tmuxTail").textContent = control.tmux_tail || "";

      renderActionEditor(state);
      updateLiveVideo(state);
      updateObservationImages(state);
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});

      currentChunkItems = buildChunkItems(state);
      syncSelectedChunk();
      updateControlButtons(phase, !!extractPlan(state));
    }}

    document.getElementById("launchForm").addEventListener("submit", function(event) {{
      event.preventDefault();
    }});

    document.getElementById("setTaskBtn").addEventListener("click", async function() {{
      const statusEl = document.getElementById("launchStatus");
      const liberoType = document.getElementById("liberoType").value;
      const suiteName = document.getElementById("suiteName").value;
      const rawTaskId = document.getElementById("taskId").value.trim();
      if (!rawTaskId) {{
        statusEl.textContent = "Set Task failed: enter a Task ID first.";
        return;
      }}
      const taskId = Number(rawTaskId);
      if (!Number.isFinite(taskId)) {{
        statusEl.textContent = "Set Task failed: Task ID must be an integer.";
        return;
      }}
      const trialIdx = Number(document.getElementById("trialIdx").value);
      const modelPath = document.getElementById("modelPath").value.trim();
      const currentModelPath = (lastState && lastState.control && lastState.control.current_model_path) || "";
      const modelChanged = !!modelPath && modelPath !== currentModelPath;
      statusEl.textContent = modelChanged ? "Reloading worker with the selected checkpoint..." : "Updating task selection...";
      try {{
        await postJson("/api/set_task", {{
          libero_type: liberoType,
          suite_name: suiteName,
          task_id: taskId,
          trial_idx: trialIdx,
          model_path: modelPath,
        }});
        followLatestChunk = true;
        selectedChunkIndex = null;
        sequencePlaying = false;
        sequenceQueue = [];
        sequenceQueueIndex = -1;
        editableActions = null;
        editableSourceActions = null;
        editableChunkKey = null;
        statusEl.textContent = modelChanged
          ? "Checkpoint reload requested. Waiting for the selected scene..."
          : "Task update sent. Waiting for the selected scene...";
      }} catch (err) {{
        statusEl.textContent = "Set Task failed: " + err.message;
      }}
    }});

    document.getElementById("resetRunBtn").addEventListener("click", async function() {{
      const statusEl = document.getElementById("launchStatus");
      statusEl.textContent = "Resetting the current run and clearing dashboard history...";
      try {{
        await postJson("/api/reset", {{}});
        followLatestChunk = true;
        selectedChunkIndex = null;
        sequencePlaying = false;
        sequenceQueue = [];
        sequenceQueueIndex = -1;
        previewVideoVersion = null;
        previewGifVersion = null;
        liveVideoVersion = null;
        liveVideoPathValue = null;
        observationImageVersion = null;
        observationImagePathValue = null;
        wristObservationImageVersion = null;
        wristObservationImagePathValue = null;
        editableActions = null;
        editableSourceActions = null;
        editableChunkKey = null;
        lastRenderedPlan = null;
        lastActionEditorRenderKey = null;
        lastState = null;
        document.getElementById("taskId").value = "";
        statusEl.textContent = "Reset complete. Previous history was cleared from the dashboard.";
        await refresh();
      }} catch (err) {{
        statusEl.textContent = "Reset failed: " + err.message;
      }}
    }});

    document.getElementById("simulateBtn").addEventListener("click", async function() {{
      const statusEl = document.getElementById("launchStatus");
      try {{
        const actions = readEditableActions();
        const chunkIdx = lastRenderedPlan ? lastRenderedPlan.chunkIdx : null;
        await postJson("/api/simulate", {{
          actions: actions,
          chunk_idx: chunkIdx,
        }});
        statusEl.textContent = "Dry-run simulation started for the edited chunk.";
      }} catch (err) {{
        statusEl.textContent = "Simulation failed: " + err.message;
      }}
    }});

    document.getElementById("runChunkBtn").addEventListener("click", async function() {{
      const statusEl = document.getElementById("launchStatus");
      try {{
        const actions = readEditableActions();
        const chunkIdx = lastRenderedPlan ? lastRenderedPlan.chunkIdx : null;
        await postJson("/api/run_chunk", {{
          actions: actions,
          chunk_idx: chunkIdx,
        }});
        statusEl.textContent = "Run Chunk started on the real environment.";
      }} catch (err) {{
        statusEl.textContent = "Run Chunk failed: " + err.message;
      }}
    }});

    document.getElementById("resetEditsBtn").addEventListener("click", function() {{
      const statusEl = document.getElementById("launchStatus");
      if (!editableSourceActions || !lastState) {{
        statusEl.textContent = "There is no editable chunk to reset yet.";
        return;
      }}
      editableActions = deepCopyActions(editableSourceActions);
      lastActionEditorRenderKey = null;
      renderActionEditor(lastState);
      statusEl.textContent = "Action editor reset to the model's original chunk output.";
    }});

    document.getElementById("liberoType").addEventListener("change", function() {{
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});
    }});

    document.getElementById("suiteName").addEventListener("change", function() {{
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});
    }});

    document.getElementById("taskId").addEventListener("input", function() {{
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});
    }});

    document.getElementById("prevChunkBtn").addEventListener("click", function() {{
      if (!currentChunkItems.length) return;
      followLatestChunk = false;
      sequencePlaying = false;
      selectedChunkIndex = selectedChunkIndex === null ? 0 : Math.max(0, selectedChunkIndex - 1);
      renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      renderChunkList();
    }});

    document.getElementById("nextChunkBtn").addEventListener("click", function() {{
      if (!currentChunkItems.length) return;
      followLatestChunk = false;
      sequencePlaying = false;
      selectedChunkIndex = selectedChunkIndex === null
        ? currentChunkItems.length - 1
        : Math.min(currentChunkItems.length - 1, selectedChunkIndex + 1);
      renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      renderChunkList();
    }});

    document.getElementById("latestChunkBtn").addEventListener("click", function() {{
      if (!currentChunkItems.length) return;
      followLatestChunk = true;
      sequencePlaying = false;
      selectedChunkIndex = currentChunkItems.length - 1;
      renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      renderChunkList();
    }});

    document.getElementById("playSequenceBtn").addEventListener("click", function() {{
      startSequencePlayback();
    }});

    document.getElementById("stopSequenceBtn").addEventListener("click", function() {{
      stopSequencePlayback();
    }});

    document.getElementById("video").addEventListener("ended", function() {{
      if (!sequencePlaying) return;
      playSequenceAt(sequenceQueueIndex + 1);
    }});

    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>
"""
