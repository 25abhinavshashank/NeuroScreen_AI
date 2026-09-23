import { Route, Routes } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import { MarketingLayout } from "@/components/layout/MarketingLayout";
import { LandingPage } from "@/pages/LandingPage";
import { ScreeningPage } from "@/pages/ScreeningPage";
import { HistoryPage } from "@/pages/HistoryPage";
import { HistoryDetailPage } from "@/pages/HistoryDetailPage";
import { MonitoringPage } from "@/pages/MonitoringPage";

export default function App() {
  return (
    <Routes>
      <Route element={<MarketingLayout />}>
        <Route path="/" element={<LandingPage />} />
      </Route>
      <Route element={<AppLayout />}>
        <Route path="/screening" element={<ScreeningPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:id" element={<HistoryDetailPage />} />
        <Route path="/monitoring" element={<MonitoringPage />} />
      </Route>
    </Routes>
  );
}
