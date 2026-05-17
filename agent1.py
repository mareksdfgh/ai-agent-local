#!/usr/bin/env python3
"""
Autonomer AI-Agent — Einzelne Python-Datei
==========================================
- Steuert deinen Computer (Shell, Dateien, Code)
- Langzeit-Gedächtnis (SQLite)
- Selbstständige Planung (Projekte, Schritte, Prioritäten)
- Zeitgesteuerte Ausführung (Cron-Style Scheduler)
- Tool Factory (kann sich selbst neue Fähigkeiten programmieren)
- Ollama LLM Integration
"""

import os, sys, json, time, sqlite3, subprocess, threading, traceback
import importlib, importlib.util, tempfile, re, datetime
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
#  Konfiguration
# ---------------------------------------------------------------------------

CONFIG = {
    "ollama_base_url": "http://localhost:11434",
    "model": "qwen3-vl:4b",
    "db_path": os.path.join(os.path.expanduser("~"), ".ai_agent", "memory.db"),
    "tools_dir": os.path.join(os.path.expanduser("~"), ".ai_agent", "tools"),
    "max_context_messages": 20,
    "scheduler_tick_seconds": 5,
    "shell": "powershell" if sys.platform == "win32" else "bash",
}

# ---------------------------------------------------------------------------
#  Hilfsfunktionen
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Erstelle notwendige Verzeichnisse."""
    os.makedirs(os.path.dirname(CONFIG["db_path"]), exist_ok=True)
    os.makedirs(CONFIG["tools_dir"], exist_ok=True)


def _ollama_check() -> bool:
    """Prüfe ob Ollama erreichbar ist."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f'{CONFIG["ollama_base_url"]}/api/tags'), timeout=5
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def ollama_chat(messages: list[dict], model: str | None = None,
                system: str | None = None) -> str:
    """Chat mit Ollama mit direktem Streaming zur Konsole."""
    import urllib.request, urllib.error

    if not _ollama_check():
        return "[FEHLER] Ollama nicht erreichbar – starte mit: ollama serve"

    model = model or CONFIG["model"]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if system:
        payload["messages"] = [{"role": "system", "content": system}] + payload["messages"]

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{CONFIG["ollama_base_url"]}/api/chat', data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=600)
        full, buf = "", ""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    token = obj.get("message", {}).get("content", "")
                    full += token
                    if token:
                        print(token, end="", flush=True)
                    if obj.get("done"):
                        resp.close()
                        return full
                except json.JSONDecodeError:
                    continue
        resp.close()
        return full or "[FEHLER] Leere Antwort"
    except Exception as exc:
        return f"[FEHLER] {exc}"


def ollama_generate(prompt: str, model: str | None = None) -> str:
    """Einfache Generierung mit Ollama (nicht Chat-basiert)."""
    import urllib.request, urllib.error

    if not _ollama_check():
        return "[FEHLER] Ollama nicht erreichbar – starte mit: ollama serve"

    model = model or CONFIG["model"]
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{CONFIG["ollama_base_url"]}/api/generate', data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=600)
        full, buf = "", ""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    token = obj.get("response", "")
                    full += token
                    if token:
                        print(token, end="", flush=True)
                    if obj.get("done"):
                        resp.close()
                        return full
                except json.JSONDecodeError:
                    continue
        resp.close()
        return full or "[FEHLER] Leere Antwort"
    except Exception as exc:
        return f"[FEHLER] {exc}"


# ---------------------------------------------------------------------------
#  MEMORY
# ---------------------------------------------------------------------------

class Memory:
    """
    Langzeit-Gedächtnis mit SQLite.

    Speichert:
    - Fakten (Category/Key/Value)
    - Konversationen (Session-basiert)
    - Projekte und Aufgaben
    - Muster und Regeln
    - Zeitgesteuerte Jobs
    """

    def __init__(self, db_path: str | None = None):
        """Initialisiere Memory mit optionalem DB-Pfad."""
        self.db_path = db_path or CONFIG["db_path"]
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Hole oder erstelle Datenbankverbindung."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        """Initialisiere Datenbankschema."""
        self._get_conn().executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(category, key)
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                timestamp TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL, description TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER, title TEXT NOT NULL, description TEXT,
                priority INTEGER DEFAULT 5, status TEXT DEFAULT 'pending',
                depends_on TEXT, created_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL, trigger_text TEXT,
                action TEXT NOT NULL, confidence REAL DEFAULT 0.5,
                times_used INTEGER DEFAULT 0, created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS scheduled_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, command TEXT NOT NULL,
                cron_expr TEXT, next_run TEXT, repeat INTEGER DEFAULT 0,
                enabled INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        self._get_conn().commit()

    def set_fact(self, category, key, value):
        """Speichere oder aktualisiere einen Fakt."""
        self._get_conn().execute(
            "INSERT INTO facts(category,key,value) VALUES(?,?,?) ON CONFLICT(category,key) DO UPDATE SET value=excluded.value,updated_at=datetime('now')",
            (category, key, value)); self._get_conn().commit()

    def get_fact(self, category, key):
        """Hole einen Fakt ab."""
        r = self._get_conn().execute("SELECT value FROM facts WHERE category=? AND key=?", (category, key)).fetchone()
        return r["value"] if r else None

    def search_facts(self, q):
        """Suche Fakten nach Query."""
        return [dict(r) for r in self._get_conn().execute(
            "SELECT category,key,value FROM facts WHERE key LIKE ? OR value LIKE ?", (f"%{q}%", f"%{q}%")).fetchall()]

    def get_all_facts(self, category=None):
        """Hole alle Fakten oder von einer Kategorie."""
        if category:
            return [dict(r) for r in self._get_conn().execute("SELECT category,key,value FROM facts WHERE category=?", (category,)).fetchall()]
        return [dict(r) for r in self._get_conn().execute("SELECT category,key,value FROM facts").fetchall()]

    def save_message(self, sid, role, content):
        """Speichere eine Nachricht."""
        self._get_conn().execute("INSERT INTO conversations(session_id,role,content) VALUES(?,?,?)", (sid, role, content))
        self._get_conn().commit()

    def get_conversation(self, sid, limit=50):
        """Hole Konversationsverlauf."""
        return [dict(r) for r in reversed(self._get_conn().execute(
            "SELECT role,content,timestamp FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?", (sid, limit)).fetchall())]

    def create_project(self, name, description=""):
        """Erstelle ein neues Projekt."""
        cur = self._get_conn().execute("INSERT OR IGNORE INTO projects(name,description) VALUES(?,?)", (name, description))
        self._get_conn().commit()
        if cur.lastrowid: return cur.lastrowid
        return self._get_conn().execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()["id"]

    def get_projects(self, status="active"):
        """Hole Projekte mit Status."""
        return [dict(r) for r in self._get_conn().execute("SELECT * FROM projects WHERE status=? ORDER BY updated_at DESC", (status,)).fetchall()]

    def update_project_status(self, name, status):
        """Aktualisiere Projektstatus."""
        self._get_conn().execute("UPDATE projects SET status=?,updated_at=datetime('now') WHERE name=?", (status, name)); self._get_conn().commit()

    def add_task(self, project_id, title, description="", priority=5, depends_on=""):
        """Füge eine Aufgabe zu Projekt hinzu."""
        cur = self._get_conn().execute("INSERT INTO tasks(project_id,title,description,priority,depends_on) VALUES(?,?,?,?,?)",
            (project_id, title, description, priority, depends_on)); self._get_conn().commit(); return cur.lastrowid

    def get_tasks(self, project_id=None, status=None):
        """Hole Aufgaben gefiltert nach Projekt und Status."""
        q, p = "SELECT * FROM tasks WHERE 1=1", []
        if project_id is not None: q += " AND project_id=?"; p.append(project_id)
        if status: q += " AND status=?"; p.append(status)
        return [dict(r) for r in self._get_conn().execute(q + " ORDER BY priority ASC, id ASC", p).fetchall()]

    def update_task_status(self, tid, status):
        """Aktualisiere Status einer Aufgabe."""
        extra = ",completed_at=datetime('now')" if status == "done" else ""
        self._get_conn().execute(f"UPDATE tasks SET status=?{extra} WHERE id=?", (status, tid)); self._get_conn().commit()

    def add_pattern(self, pt, trigger, action, conf=0.5):
        """Füge ein Muster/Regel hinzu."""
        self._get_conn().execute("INSERT INTO patterns(pattern_type,trigger_text,action,confidence) VALUES(?,?,?,?)", (pt, trigger, action, conf)); self._get_conn().commit()

    def find_patterns(self, text):
        """Finde Muster für Text."""
        return [dict(r) for r in self._get_conn().execute("SELECT * FROM patterns WHERE trigger_text LIKE ? ORDER BY confidence DESC", (f"%{text}%",)).fetchall()]

    def add_scheduled_job(self, name, command, cron_expr="", next_run="", repeat=0):
        """Erstelle einen zeitgesteuerten Job."""
        cur = self._get_conn().execute("INSERT INTO scheduled_jobs(name,command,cron_expr,next_run,repeat) VALUES(?,?,?,?,?)",
            (name, command, cron_expr, next_run, repeat)); self._get_conn().commit(); return cur.lastrowid

    def get_due_jobs(self):
        """Hole fällige Jobs."""
        return [dict(r) for r in self._get_conn().execute(
            "SELECT * FROM scheduled_jobs WHERE enabled=1 AND next_run<=? ORDER BY next_run ASC", (datetime.datetime.now().isoformat(),)).fetchall()]

    def get_all_jobs(self):
        """Hole alle aktiven Jobs."""
        return [dict(r) for r in self._get_conn().execute("SELECT * FROM scheduled_jobs WHERE enabled=1 ORDER BY next_run ASC").fetchall()]

    def update_job_next_run(self, jid, nr):
        """Aktualisiere nächste Laufzeit eines Jobs."""
        self._get_conn().execute("UPDATE scheduled_jobs SET next_run=? WHERE id=?", (nr, jid)); self._get_conn().commit()

    def disable_job(self, jid):
        """Deaktiviere einen Job."""
        self._get_conn().execute("UPDATE scheduled_jobs SET enabled=0 WHERE id=?", (jid,)); self._get_conn().commit()

    def close(self):
        """Schließe Datenbankverbindung."""
        if self._conn: self._conn.close(); self._conn = None


# ---------------------------------------------------------------------------
#  EXECUTOR
# ---------------------------------------------------------------------------

class Executor:
    """
    Führt Befehle aus (Shell, Python, Dateien).

    Methoden:
    - run_shell(): PowerShell/Bash
    - run_python(): Python-Code
    - read_file(): Datei lesen
    - write_file(): Datei schreiben
    - list_directory(): Verzeichnis auflisten
    """

    @staticmethod
    def run_shell(command, timeout=None):
        """Führe Shell-Befehl aus — volle Macht, kein Limit."""
        try:
            args = ["powershell", "-NoProfile", "-Command", command] if sys.platform == "win32" else ["bash", "-c", command]
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout if timeout else None, cwd=os.getcwd())
            return {"success": r.returncode == 0, "stdout": r.stdout.strip(), "stderr": r.stderr.strip(), "returncode": r.returncode}
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"Timeout nach {timeout}s", "returncode": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    @staticmethod
    def run_python(code, timeout=None):
        """Führe Python-Code aus — kein Zeitlimit."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
        try:
            tmp.write(code); tmp.close()
            r = subprocess.run([sys.executable, tmp.name], capture_output=True, text=True, timeout=timeout if timeout else None)
            return {"success": r.returncode == 0, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"Timeout nach {timeout}s"}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}
        finally:
            os.unlink(tmp.name)

    @staticmethod
    def read_file(path):
        """Lese Datei."""
        try:
            p = Path(path).expanduser().resolve()
            return {"success": True, "content": p.read_text(encoding="utf-8"), "path": str(p)}
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}

    @staticmethod
    def write_file(path, content):
        """Schreibe Datei."""
        try:
            p = Path(path).expanduser().resolve(); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8"); return {"success": True, "path": str(p)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def list_directory(path="."):
        """Liste Verzeichnis auf."""
        try:
            p = Path(path).expanduser().resolve()
            entries = [{"name": e.name, "type": "dir" if e.is_dir() else "file",
                        "size": e.stat().st_size if e.is_file() else 0} for e in sorted(p.iterdir())]
            return {"success": True, "entries": entries, "path": str(p)}
        except Exception as e:
            return {"success": False, "entries": [], "error": str(e)}


# ---------------------------------------------------------------------------
#  TOOL FACTORY
# ---------------------------------------------------------------------------

class ToolFactory:
    """
    Lädt und verwaltet Custom-Tools.

    Tools sind Python-Dateien mit run/execute/main-Funktion.
    Agent kann neue Tools programmieren und ausführen.
    """

    def __init__(self, tools_dir=None):
        """Initialisiere Tool Factory."""
        self.tools_dir = tools_dir or CONFIG["tools_dir"]
        os.makedirs(self.tools_dir, exist_ok=True)
        self._loaded: dict[str, Callable] = {}
        self._load_all()

    def _load_all(self):
        """Lade alle Tools aus Verzeichnis."""
        for f in Path(self.tools_dir).glob("*.py"):
            try: self._load_file(f)
            except: pass

    def _load_file(self, path):
        """Lade eine Tool-Datei."""
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            for fn in ("run", "execute", "main"):
                if hasattr(mod, fn): self._loaded[path.stem] = getattr(mod, fn); return

    def create_tool(self, name, code, description=""):
        """Erstelle ein neues Tool."""
        path = Path(self.tools_dir) / f"{name}.py"
        full = f'"""\nTool: {name}\n{description}\n"""\n\n' + code
        try: compile(full, str(path), "exec")
        except SyntaxError as e: return {"success": False, "error": str(e)}
        path.write_text(full, encoding="utf-8")
        try: self._load_file(path); return {"success": True, "name": name, "path": str(path)}
        except Exception as e: return {"success": False, "error": str(e)}

    def run_tool(self, name, *a, **kw):
        """Führe ein Tool aus."""
        if name not in self._loaded: return {"success": False, "error": f"Tool '{name}' nicht gefunden"}
        try: return {"success": True, "result": self._loaded[name](*a, **kw)}
        except Exception as e: return {"success": False, "error": str(e)}

    def list_tools(self):
        """Liste alle geladenen Tools auf."""
        return list(self._loaded.keys())


# ---------------------------------------------------------------------------
#  PLANNER
# ---------------------------------------------------------------------------

class Planner:
    """
    Plant Aufgaben und verwaltet Projekte.

    Nutzt LLM um Aufgabenbeschreibungen in Schritte zu zerlegen.
    """

    def __init__(self, memory):
        """Initialisiere Planner."""
        self.memory = memory

    def plan_task(self, description):
        """Plane eine Aufgabe (zerlege in Schritte)."""
        sys_prompt = """Du bist ein Aufgabenplaner. Zerlege die Aufgabe in Schritte.
Antworte NUR mit JSON: {"project_name":"...","description":"...","steps":[{"title":"...","description":"...","priority":1}]}"""
        print("Planing: ", end="", flush=True)
        response = ollama_chat(
            [{"role": "user", "content": f"Plane:\n{description}"}],
            system=sys_prompt)
        print()
        try:
            m = re.search(r'\{[\s\S]*\}', response)
            if not m: return {"success": False, "error": "Kein JSON", "raw": response}
            plan = json.loads(m.group())
        except: return {"success": False, "error": "JSON-Fehler", "raw": response}
        pid = self.memory.create_project(plan.get("project_name", "Unbenannt"), plan.get("description", ""))
        ids = []
        for s in plan.get("steps", []):
            ids.append(self.memory.add_task(pid, s.get("title", ""), s.get("description", ""), s.get("priority", 5), s.get("depends_on", "")))
        return {"success": True, "project_id": pid, "project_name": plan.get("project_name"), "task_count": len(ids), "tasks": plan.get("steps", [])}

    def get_next_task(self, project_id=None):
        """Hole nächste Aufgabe."""
        tasks = self.memory.get_tasks(project_id=project_id, status="pending")
        if not tasks: return None
        done_ids = {str(t["id"]) for t in self.memory.get_tasks(project_id=project_id, status="done")}
        for t in tasks:
            deps = t.get("depends_on", "")
            if not deps or all(d.strip() in done_ids for d in deps.split(",")): return t
        return tasks[0]

    def get_project_status(self, pid):
        """Hole Status eines Projekts."""
        tasks = self.memory.get_tasks(project_id=pid)
        d = sum(1 for t in tasks if t["status"] == "done")
        return {"total": len(tasks), "done": d, "pending": sum(1 for t in tasks if t["status"] == "pending"),
                "in_progress": sum(1 for t in tasks if t["status"] == "in_progress"),
                "progress_pct": round(d / len(tasks) * 100, 1) if tasks else 0}


# ---------------------------------------------------------------------------
#  SCHEDULER
# ---------------------------------------------------------------------------

class CronParser:
    """
    Parser für Cron-Ausdrücke (Subset).

    Format: "minute hour day month weekday"
    Beispiele:
    - "*/5 * * * *"  (alle 5 Minuten)
    - "0 9 * * 1-5"  (Mo-Fr um 9:00)
    - "30 14 15 * *"  (jeden 15. um 14:30)
    """

    @staticmethod
    def parse_field(f, lo, hi):
        """Parse ein Cron-Feld."""
        vals = set()
        for p in f.split(","):
            p = p.strip()
            if p == "*": vals.update(range(lo, hi + 1))
            elif p.startswith("*/"): vals.update(range(lo, hi + 1, int(p[2:])))
            elif "-" in p: a, b = p.split("-"); vals.update(range(int(a), int(b) + 1))
            else: vals.add(int(p))
        return vals

    @staticmethod
    def matches(expr, dt=None):
        """Prüfe ob Ausdruck auf Datum zutrifft."""
        if not expr: return False
        dt = dt or datetime.datetime.now(); parts = expr.split()
        if len(parts) != 5: return False
        for e, lo, hi, cur in zip(parts, [0,0,1,1,0], [59,23,31,12,6], [dt.minute,dt.hour,dt.day,dt.month,dt.weekday()]):
            if cur not in CronParser.parse_field(e, lo, hi): return False
        return True

    @staticmethod
    def next_run(expr, dt=None):
        """Berechne nächste Laufzeit."""
        dt = (dt or datetime.datetime.now()).replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        for _ in range(525960):
            if CronParser.matches(expr, dt): return dt.isoformat()
            dt += datetime.timedelta(minutes=1)
        return ""


class Scheduler:
    """
    Zeitgesteuerte Job-Ausführung.

    Läuft im Hintergrund und führt fällige Jobs aus.
    Unterstützt Cron-Ausdrücke und Wiederholung.
    """

    def __init__(self, memory, executor):
        """Initialisiere Scheduler."""
        self.memory, self.executor = memory, executor
        self._running = False; self._results = []

    def add_job(self, name, command, cron_expr="", run_at="", repeat=0):
        """Füge einen Job hinzu."""
        nr = CronParser.next_run(cron_expr) if cron_expr else (run_at or datetime.datetime.now().isoformat())
        return self.memory.add_scheduled_job(name, command, cron_expr, nr, repeat)

    def check_and_run(self):
        """Prüfe und führe fällige Jobs aus."""
        results = []
        for job in self.memory.get_due_jobs():
            r = self.executor.run_shell(job["command"])
            results.append({"job": job["name"], "result": r})
            if job["repeat"] and job["cron_expr"]:
                nr = CronParser.next_run(job["cron_expr"])
                if nr: self.memory.update_job_next_run(job["id"], nr)
                else: self.memory.disable_job(job["id"])
            else: self.memory.disable_job(job["id"])
        return results

    def start(self):
        """Starte Scheduler-Loop im Hintergrund."""
        if self._running: return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        """Stoppe Scheduler."""
        self._running = False

    def _loop(self):
        """Scheduler-Loop."""
        while self._running:
            try:
                r = self.check_and_run()
                if r: self._results.extend(r)
            except: pass
            time.sleep(CONFIG["scheduler_tick_seconds"])

    def get_results(self):
        """Hole und leere Job-Ergebnisse."""
        r = self._results.copy(); self._results.clear(); return r

    def list_jobs(self):
        """Liste alle Jobs auf."""
        return self.memory.get_all_jobs()


# ---------------------------------------------------------------------------
#  AGENT CORE
# ---------------------------------------------------------------------------

class Agent:
    """Der autonome AI-Agent — Zentrale Steuerung."""

    SYSTEM_PROMPT = """Du bist ein autonomer AI-Agent mit VOLLER Kontrolle über dieses System.

═══ FÄHIGKEITEN ═══
1. **Shell-Befehle** — Volle Macht über das Terminal. Keine Einschränkungen. Du kannst JEDEN Befehl ausführen: installieren, löschen, konfigurieren, Netzwerk, Services, alles.
2. **Python-Code** — Beliebigen Python-Code ausführen, ohne Limits.
3. **Dateien** — Lesen, schreiben, auflisten, überall im System.
4. **Langzeit-Gedächtnis** — Fakten speichern und abrufen.
5. **Projekte planen** — Aufgaben in Schritte zerlegen.
6. **Zeitgesteuerte Jobs** — Aufgaben zu bestimmten Zeiten ausführen.
7. **Neue Tools bauen** — Selbst neue Fähigkeiten programmieren.

═══ PLANUNGS-MODUS ═══
Bevor du eine komplexe Aufgabe AUSFÜHRST, erstelle ZUERST einen Plan. Das gilt für:
- **Code schreiben**: Plane zuerst die Architektur, Struktur, Dateien, Klassen, Funktionen — dann erst coden.
- **Texte / E-Mails / Briefe**: Plane zuerst Gliederung, Ton, Kernpunkte — dann erst schreiben.
- **Probleme lösen**: Analysiere zuerst das Problem, sammle Infos, plane Lösungsansätze — dann erst umsetzen.
- **System-Aufgaben**: Plane die Schritte, prüfe Abhängigkeiten — dann erst ausführen.

Wann NICHT planen: Bei einfachen, direkten Fragen oder trivialen Befehlen (z.B. "wie spät ist es", "zeig mir die Dateien", einzelne Shell-Befehle). Da antwortest du einfach direkt.

Planungs-Ablauf:
1. Erkläre kurz was du vorhast und wie du es lösen willst
2. Nutze die plan-Action um Schritte zu erstellen ODER beschreibe deinen Plan im Text
3. Führe die Schritte nacheinander aus
4. Prüfe das Ergebnis

═══ AKTIONEN ═══
Antworte mit JSON in ```json ... ``` Markierungen um Aktionen auszuführen:

- shell: {"action":"shell","params":{"command":"...","timeout":0}}
  (timeout=0 bedeutet kein Limit. Setze einen Wert in Sekunden wenn nötig.)
- python: {"action":"python","params":{"code":"..."}}
- read_file: {"action":"read_file","params":{"path":"..."}}
- write_file: {"action":"write_file","params":{"path":"...","content":"..."}}
- list_dir: {"action":"list_dir","params":{"path":"."}}
- remember: {"action":"remember","params":{"category":"...","key":"...","value":"..."}}
- recall: {"action":"recall","params":{"category":"...","key":"..."}}
- search_memory: {"action":"search_memory","params":{"query":"..."}}
- plan: {"action":"plan","params":{"description":"..."}}
- next_task: {"action":"next_task","params":{}}
- complete_task: {"action":"complete_task","params":{"task_id":1}}
- show_projects: {"action":"show_projects","params":{}}
- show_tasks: {"action":"show_tasks","params":{"project_id":1}}
- schedule: {"action":"schedule","params":{"name":"...","command":"...","cron":"...","repeat":1}}
- show_jobs: {"action":"show_jobs","params":{}}
- create_tool: {"action":"create_tool","params":{"name":"...","code":"...","description":"..."}}
- use_tool: {"action":"use_tool","params":{"name":"...","args":[],"kwargs":{}}}
- list_tools: {"action":"list_tools","params":{}}
- multi: {"action":"multi","params":{"actions":[...]}}

Ohne JSON-Block antwortest du nur mit Text.
Du hast volle Systemrechte. Sei proaktiv, plane komplexe Aufgaben durch, speichere wichtige Infos im Gedächtnis. Handle autonom und entschlossen."""

    def __init__(self):
        """Initialisiere Agent."""
        ensure_dirs()
        self.memory = Memory(); self.executor = Executor()
        self.tool_factory = ToolFactory(); self.planner = Planner(self.memory)
        self.scheduler = Scheduler(self.memory, self.executor)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation: list[dict] = []
        self._lock = threading.Lock()

    def _build_system(self):
        """Baue System-Prompt mit Kontext."""
        parts = [self.SYSTEM_PROMPT]
        facts = self.memory.get_all_facts()
        if facts:
            parts.append("\nBekannte Fakten:")
            for f in facts[:30]: parts.append(f"  [{f['category']}] {f['key']}: {f['value']}")
        projects = self.memory.get_projects()
        if projects:
            parts.append("\nAktive Projekte:")
            for p in projects[:5]:
                s = self.planner.get_project_status(p["id"])
                parts.append(f"  - {p['name']}: {s['done']}/{s['total']} erledigt")
        tools = self.tool_factory.list_tools()
        if tools: parts.append(f"\nCustom-Tools: {', '.join(tools)}")
        parts.append(f"\nDatum: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        parts.append(f"CWD: {os.getcwd()}")
        parts.append(f"OS: {sys.platform}")
        return "\n".join(parts)

    def _exec(self, ad):
        """Führe eine Action aus."""
        a, p = ad.get("action", ""), ad.get("params", {})
        if a == "shell":
            t = p.get("timeout", None)
            r = self.executor.run_shell(p.get("command", ""), timeout=t if t else None)
            o = r["stdout"]
            if r["stderr"]: o += f"\n[STDERR] {r['stderr']}"
            return f"[Shell {'OK' if r['success'] else 'FAIL'}] {o}"
        elif a == "python":
            r = self.executor.run_python(p.get("code", ""))
            o = r["stdout"]
            if r["stderr"]: o += f"\n[STDERR] {r['stderr']}"
            return f"[Python {'OK' if r['success'] else 'FAIL'}] {o}"
        elif a == "read_file":
            r = self.executor.read_file(p.get("path", ""))
            return f"[Datei: {r.get('path','')}]\n{r['content'][:5000]}" if r["success"] else f"[Fehler] {r.get('error')}"
        elif a == "write_file":
            r = self.executor.write_file(p.get("path", ""), p.get("content", ""))
            return f"[Geschrieben: {r.get('path','')}]" if r["success"] else f"[Fehler] {r.get('error')}"
        elif a == "list_dir":
            r = self.executor.list_directory(p.get("path", "."))
            if r["success"]:
                lines = [f"[{r['path']}] ({len(r['entries'])} Einträge)"]
                for e in r["entries"][:80]:
                    lines.append(f"  {'DIR' if e['type']=='dir' else 'FILE'} {e['name']}")
                return "\n".join(lines)
            return f"[Fehler] {r.get('error')}"
        elif a == "remember":
            self.memory.set_fact(p.get("category",""), p.get("key",""), p.get("value",""))
            return f"[Gespeichert] {p.get('category')}/{p.get('key')}"
        elif a == "recall":
            v = self.memory.get_fact(p.get("category",""), p.get("key",""))
            return f"[Memory] {v}" if v else "[Nicht gefunden]"
        elif a == "search_memory":
            rs = self.memory.search_facts(p.get("query",""))
            if rs: return "\n".join([f"  [{r['category']}] {r['key']}: {r['value']}" for r in rs])
            return "[Keine Ergebnisse]"
        elif a == "plan":
            r = self.planner.plan_task(p.get("description",""))
            if r["success"]:
                return f"[Projekt: {r['project_name']}] {r['task_count']} Schritte erstellt"
            return f"[Planungsfehler] {r.get('error','')}"
        elif a == "next_task":
            t = self.planner.get_next_task(p.get("project_id"))
            if t: self.memory.update_task_status(t["id"], "in_progress"); return f"[Task #{t['id']}] P{t['priority']}: {t['title']}"
            return "[Keine Aufgaben]"
        elif a == "complete_task":
            self.memory.update_task_status(p.get("task_id"), "done"); return f"[Task #{p.get('task_id')} erledigt]"
        elif a == "show_projects":
            ps = self.memory.get_projects()
            if not ps: return "[Keine Projekte]"
            return "\n".join([f"  #{p['id']} {p['name']} ({self.planner.get_project_status(p['id'])['progress_pct']}%)" for p in ps])
        elif a == "show_tasks":
            ts = self.memory.get_tasks(project_id=p.get("project_id"))
            if not ts: return "[Keine Tasks]"
            icons = {"pending": "WAIT", "in_progress": "RUN", "done": "DONE"}
            return "\n".join([f"  [{icons.get(t['status'],'?')}] #{t['id']} P{t['priority']}: {t['title']}" for t in ts])
        elif a == "show_jobs":
            js = self.scheduler.list_jobs()
            if not js: return "[Keine Jobs]"
            return "\n".join([f"  #{j['id']} {j['name']} next:{j['next_run']}" for j in js])
        elif a == "schedule":
            jid = self.scheduler.add_job(p.get("name",""), p.get("command",""), p.get("cron",""), p.get("run_at",""), p.get("repeat",0))
            return f"[Job #{jid} geplant]"
        elif a == "create_tool":
            r = self.tool_factory.create_tool(p.get("name",""), p.get("code",""), p.get("description",""))
            return f"[Tool erstellt: {r.get('name','')}]" if r["success"] else f"[Tool-Fehler] {r.get('error')}"
        elif a == "use_tool":
            r = self.tool_factory.run_tool(p.get("name",""), *p.get("args",[]), **p.get("kwargs",{}))
            return f"[Tool] {r.get('result','')}" if r["success"] else f"[Tool-Fehler] {r.get('error')}"
        elif a == "list_tools":
            t = self.tool_factory.list_tools()
            return f"[Tools] {', '.join(t)}" if t else "[Keine Tools]"
        elif a == "multi":
            return "\n---\n".join([self._exec(sa) for sa in p.get("actions",[])])
        return f"[Unbekannte Aktion: {a}]"

    def _extract_actions(self, text):
        """Extrahiere JSON-Actions aus Antwort."""
        actions = []
        for m in re.findall(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL):
            try:
                d = json.loads(m.strip())
                if isinstance(d, dict) and "action" in d: actions.append(d)
            except: pass
        if not actions:
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith('{"action"'):
                    try:
                        d = json.loads(line)
                        if "action" in d: actions.append(d)
                    except: pass
        return actions

    def _strip_actions(self, text):
        """Entferne JSON-Blöcke aus Text."""
        c = re.sub(r'```json\s*\n?.*?\n?```', '', text, flags=re.DOTALL)
        return re.sub(r'\n{3,}', '\n\n', c).strip()

    def process_message(self, user_input):
        """Verarbeite eine Benutzernachricht."""
        with self._lock:
            self.memory.save_message(self.session_id, "user", user_input)
            self.conversation.append({"role": "user", "content": user_input})
            if len(self.conversation) > CONFIG["max_context_messages"]:
                self.conversation = self.conversation[-CONFIG["max_context_messages"]:]

            sys_prompt = self._build_system()
            print("\nAgent: ", end="", flush=True)
            response = ollama_chat(self.conversation, system=sys_prompt)

            if response.startswith("[FEHLER]"):
                self.conversation.append({"role": "assistant", "content": response})
                print()
                return response

            actions = self._extract_actions(response)
            action_results = []
            for ad in actions:
                print(f"\n  → {ad.get('action','')} ...", end="", flush=True)
                result = self._exec(ad)
                action_results.append(result)
                print(f"\n  ← {result[:200]}")

            visible = self._strip_actions(response)

            if action_results:
                self.conversation.append({"role": "assistant", "content": response})
                self.conversation.append({"role": "user", "content": f"[System: Aktionsergebnisse]\n" + "\n\n".join(action_results)})

                print("\nAgent: ", end="", flush=True)
                followup = ollama_chat(self.conversation, system=sys_prompt)
                follow_actions = self._extract_actions(followup)

                iteration = 0
                while follow_actions and iteration < 5:
                    iteration += 1
                    for fa in follow_actions:
                        print(f"\n  → {fa.get('action','')} ...", end="", flush=True)
                        r = self._exec(fa)
                        action_results.append(r)
                        print(f"\n  ← {r[:200]}")
                    self.conversation.append({"role": "assistant", "content": followup})
                    self.conversation.append({"role": "user", "content": f"[System: Aktionsergebnisse]\n" + "\n\n".join(action_results[-len(follow_actions):])})
                    print("\nAgent: ", end="", flush=True)
                    followup = ollama_chat(self.conversation, system=sys_prompt)
                    follow_actions = self._extract_actions(followup)

                visible_f = self._strip_actions(followup)
                full = f"{visible}\n\n{visible_f}".strip() if visible else visible_f
                self.conversation.append({"role": "assistant", "content": followup})
            else:
                full = visible
                self.conversation.append({"role": "assistant", "content": response})

            print()
            self.memory.save_message(self.session_id, "assistant", full)
            return full

    def get_status(self):
        """Hole Agent-Status."""
        projects = self.memory.get_projects()
        proj_data = []
        for p in projects:
            s = self.planner.get_project_status(p["id"])
            tasks = self.memory.get_tasks(project_id=p["id"])
            proj_data.append({**p, "status_info": s, "tasks": tasks})
        return {
            "session": self.session_id,
            "model": CONFIG["model"],
            "projects": proj_data,
            "facts": self.memory.get_all_facts(),
            "tools": self.tool_factory.list_tools(),
            "jobs": self.scheduler.list_jobs(),
            "conversation_count": len(self.conversation),
        }

    def run_interactive(self):
        """Starte interaktive CLI-Loop."""
        print("=" * 60)
        print(f"  Autonomer AI-Agent · {CONFIG['model']}")
        print("=" * 60)
        print("\nBefehle:")
        print("  /help       - Zeige diese Hilfe")
        print("  /status     - Zeige Agent-Status")
        print("  /memory     - Zeige Gedächtnis")
        print("  /projects   - Zeige Projekte")
        print("  /tasks      - Zeige Aufgaben")
        print("  /tools      - Zeige Custom-Tools")
        print("  /jobs       - Zeige geplante Jobs")
        print("  /history    - Zeige Konversationsverlauf")
        print("  /clear      - Starte neuen Chat")
        print("  /model NAME - Wechsle Modell")
        print("  /quit       - Beende Agent")
        print()

        while True:
            try:
                inp = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not inp: continue

            if inp == "/quit": break

            if inp == "/help":
                print("\nBefehle:")
                print("  /help       - Diese Hilfe")
                print("  /status     - Agent-Status")
                print("  /memory     - Gedächtnis")
                print("  /projects   - Projekte")
                print("  /tasks      - Aufgaben")
                print("  /tools      - Custom-Tools")
                print("  /jobs       - Geplante Jobs")
                print("  /history    - Konversationsverlauf")
                print("  /clear      - Neuer Chat")
                print("  /model NAME - Modell wechseln")
                print("  /quit       - Beenden")
                print()
                continue

            if inp == "/status":
                s = self.get_status()
                print(f"\nProjekte: {len(s['projects'])} | Fakten: {len(s['facts'])} | Tools: {len(s['tools'])} | Jobs: {len(s['jobs'])}")
                print(f"Konversation: {s['conversation_count']} Nachrichten | Modell: {s['model']}\n")
                continue

            if inp == "/memory":
                facts = self.memory.get_all_facts()
                if not facts:
                    print("\nGedächtnis ist leer.\n")
                else:
                    print("\nGedächtnis:")
                    for f in facts:
                        print(f"  [{f['category']}] {f['key']}: {f['value']}")
                    print()
                continue

            if inp == "/projects":
                projects = self.memory.get_projects()
                if not projects:
                    print("\nKeine Projekte.\n")
                else:
                    print("\nProjekte:")
                    for p in projects:
                        s = self.planner.get_project_status(p["id"])
                        print(f"  #{p['id']} {p['name']} - {s['done']}/{s['total']} erledigt ({s['progress_pct']}%)")
                    print()
                continue

            if inp == "/tasks":
                tasks = self.memory.get_tasks()
                if not tasks:
                    print("\nKeine Aufgaben.\n")
                else:
                    print("\nAufgaben:")
                    for t in tasks:
                        print(f"  #{t['id']} [{t['status']}] P{t['priority']}: {t['title']}")
                    print()
                continue

            if inp == "/tools":
                tools = self.tool_factory.list_tools()
                if not tools:
                    print("\nKeine Custom-Tools.\n")
                else:
                    print(f"\nCustom-Tools: {', '.join(tools)}\n")
                continue

            if inp == "/jobs":
                jobs = self.scheduler.list_jobs()
                if not jobs:
                    print("\nKeine geplanten Jobs.\n")
                else:
                    print("\nGeplante Jobs:")
                    for j in jobs:
                        print(f"  #{j['id']} {j['name']} - nächstens: {j['next_run']}")
                    print()
                continue

            if inp == "/history":
                if not self.conversation:
                    print("\nKein Verlauf.\n")
                else:
                    print("\nKonversationsverlauf:")
                    for msg in self.conversation:
                        role = "Du" if msg["role"] == "user" else "Agent"
                        content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
                        print(f"  {role}: {content}")
                    print()
                continue

            if inp == "/clear":
                self.conversation.clear()
                self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                print("\nChat geleert.\n")
                continue

            if inp.startswith("/model "):
                model = inp[7:].strip()
                CONFIG["model"] = model
                print(f"\nModell auf '{model}' gesetzt.\n")
                continue

            # Normale Nachricht verarbeiten (Streaming-Ausgabe passiert in process_message)
            self.process_message(inp)


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    """Haupteinstiegspunkt."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
Autonomer AI-Agent
==================

Aufruf:
  python agent.py                 Interaktiver Chat
  python agent.py --model NAME    Modell setzen
  python agent.py --ollama URL    Ollama-URL setzen
  python agent.py --run "TEXT"    Einzelne Nachricht

Befehle im Chat:
  /help       Zeige Hilfe
  /status     Agent-Status
  /memory     Gedächtnis anzeigen
  /projects   Projekte anzeigen
  /tasks      Aufgaben anzeigen
  /tools      Custom-Tools anzeigen
  /jobs       Geplante Jobs anzeigen
  /history    Konversationsverlauf
  /clear      Chat leeren
  /model NAME Modell wechseln
  /quit       Beenden
        """)
        return

    # Args parsen
    i = 1
    run_cmd = None
    while i < len(sys.argv):
        if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            CONFIG["model"] = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--run" and i + 1 < len(sys.argv):
            run_cmd = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--ollama" and i + 1 < len(sys.argv):
            CONFIG["ollama_base_url"] = sys.argv[i + 1]; i += 2
        else:
            i += 1

    agent = Agent()
    agent.scheduler.start()

    try:
        if run_cmd:
            agent.process_message(run_cmd)
            agent.scheduler.stop()
            agent.memory.close()
            return

        agent.run_interactive()
    except KeyboardInterrupt:
        print("\n\nBeendet.")
    finally:
        agent.scheduler.stop()
        agent.memory.close()


if __name__ == "__main__":
    main()
