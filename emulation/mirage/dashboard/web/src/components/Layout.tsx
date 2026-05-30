import { NavLink, Outlet } from "react-router-dom";

export function Layout() {
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
          <NavLink to="/sessions">Sessions</NavLink>
          <NavLink to="/topology">Topology</NavLink>
        </nav>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
