import { NavLink, Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <img src="/favicon.svg" alt="AMD Mirage" className="sidebar-logo" />
          <div>
            <h1>Mirage</h1>
            <span className="subtitle">AMD GPU Simulator</span>
          </div>
        </div>
        <nav>
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/simulators">Simulators</NavLink>
          <NavLink to="/profiles">Profiles</NavLink>
          <NavLink to="/sessions">Sessions</NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/topology">Topology Editor</NavLink>
        </nav>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
