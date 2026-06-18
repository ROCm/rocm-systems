import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ToastProvider } from "./components/ui/Toast";
import { AgentEditorPage } from "./pages/AgentEditorPage";
import { AgentsPage } from "./pages/AgentsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { ProfilesPage } from "./pages/ProfilesPage";
import { SessionsPage } from "./pages/SessionsPage";
import { SessionDetailPage } from "./pages/SessionDetailPage";
import { TopologiesPage } from "./pages/TopologiesPage";
import "./App.css";

function App() {
  return (
    <ToastProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="profiles" element={<ProfilesPage />} />
          <Route path="topologies" element={<TopologiesPage />} />
          <Route path="agents" element={<AgentsPage />} />
          <Route path="agents/new" element={<AgentEditorPage />} />
          <Route path="agents/edit/:name" element={<AgentEditorPage />} />
          <Route path="sessions" element={<SessionsPage />} />
          <Route path="sessions/:id" element={<SessionDetailPage />} />
        </Route>
      </Routes>
    </ToastProvider>
  );
}

export default App;
