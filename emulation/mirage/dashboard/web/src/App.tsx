import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { OverviewPage } from "./pages/OverviewPage";
import { SimulatorListPage } from "./pages/SimulatorListPage";
import { SimulatorDetailPage } from "./pages/SimulatorDetailPage";
import { ProfileListPage } from "./pages/ProfileListPage";
import { SessionListPage } from "./pages/SessionListPage";
import { SessionDetailPage } from "./pages/SessionDetailPage";
import { RunListPage } from "./pages/RunListPage";
import { TopologyEditorPage } from "./pages/TopologyEditorPage";
import "./App.css";

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<OverviewPage />} />
        <Route path="topology" element={<TopologyEditorPage />} />
        <Route path="simulators" element={<SimulatorListPage />} />
        <Route path="simulators/:name" element={<SimulatorDetailPage />} />
        <Route path="profiles" element={<ProfileListPage />} />
        <Route path="sessions" element={<SessionListPage />} />
        <Route path="sessions/:name" element={<SessionDetailPage />} />
        <Route path="runs" element={<RunListPage />} />
      </Route>
    </Routes>
  );
}

export default App;
