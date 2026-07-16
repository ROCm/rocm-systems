import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import * as api from "../api/client";
import type { SystemInfo } from "../api/types";

export function Layout() {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  useEffect(() => {
    api.getSystem().then(setSys).catch(() => setSys(null));
  }, []);
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div>
            <h1>Mirage</h1>
            <span className="subtitle">Dashboard</span>
          </div>
        </div>
        <nav>
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/profiles">Profiles</NavLink>
          <NavLink to="/topologies">Topologies</NavLink>
          <NavLink to="/agents">Agents</NavLink>
          <NavLink to="/sessions">Sessions</NavLink>
        </nav>
        <div className="sidebar-footer">
          {sys && (
            <>
              <div className="sidebar-version" data-testid="daemon-version">
                v{sys.daemon_version}
              </div>
              <div className="sidebar-default-emu" data-testid="default-emulator">
                default: {sys.default_emulator}
              </div>
            </>
          )}
        </div>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
