import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ToastProvider } from "./components/ui/Toast";
import { OverviewPage } from "./pages/OverviewPage";
import { ProfilesPage } from "./pages/ProfilesPage";
import { SessionsPage } from "./pages/SessionsPage";
import { SessionDetailPage } from "./pages/SessionDetailPage";
import { TopologyEditorPage } from "./pages/TopologyEditorPage";
import "./App.css";

function App() {
  return (
    <ToastProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="profiles" element={<ProfilesPage />} />
          <Route path="sessions" element={<SessionsPage />} />
          <Route path="sessions/:id" element={<SessionDetailPage />} />
          <Route path="topology" element={<TopologyEditorPage />} />
        </Route>
      </Routes>
    </ToastProvider>
  );
}

export default App;
