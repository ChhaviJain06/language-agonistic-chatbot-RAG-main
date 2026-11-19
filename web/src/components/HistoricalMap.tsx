import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { formatEra } from "../lib/text";

interface Props {
  coordinates: [number, number];
  entityName: string;
  dynastyName: string | null;
  eraYears: [number, number] | null;
  theme: "light" | "dark";
}

const FORCE_OFFLINE = import.meta.env.VITE_OFFLINE_MAP === "1";

// Approximate Indian subcontinent geographic landmass for offline vector rendering
const SUBCONTINENT_OUTLINE: [number, number][] = [
  [37.0, 74.0], [35.5, 77.0], [32.0, 79.0], [30.0, 81.0], [28.0, 89.0],
  [27.5, 97.0], [24.0, 94.0], [22.0, 92.0], [21.5, 87.0], [16.0, 82.0],
  [13.0, 80.3], [10.0, 79.8], [8.0, 77.5], [10.0, 76.0], [13.0, 74.5],
  [16.0, 73.3], [19.0, 72.8], [21.0, 72.5], [23.0, 68.5], [25.0, 67.5],
  [28.0, 66.5], [31.5, 70.0], [35.0, 71.5], [37.0, 74.0],
];

// Approximate historical empire boundaries
const EMPIRE_BOUNDARIES: Record<string, [number, number][]> = {
  "Maurya Empire": [
    [34.0, 71.0], [32.0, 76.0], [28.0, 84.0], [26.0, 90.0],
    [22.0, 89.0], [15.0, 76.0], [13.0, 78.0], [18.0, 73.0],
    [24.0, 68.0], [30.0, 67.0], [34.0, 71.0],
  ],
  "Mughal Empire": [
    [34.5, 69.0], [32.0, 77.0], [27.0, 85.0], [24.0, 91.0],
    [22.0, 88.0], [20.0, 82.0], [18.0, 74.0], [24.0, 69.0],
    [30.0, 66.0], [34.5, 69.0],
  ],
  "Chola Dynasty": [
    [14.0, 80.0], [13.0, 80.5], [10.0, 79.8], [8.0, 77.5],
    [10.0, 76.5], [12.0, 77.5], [14.0, 80.0],
  ],
  "Indus Valley Civilisation": [
    [32.0, 71.0], [30.0, 74.0], [27.0, 71.0], [23.0, 69.0],
    [24.0, 66.5], [28.0, 66.0], [32.0, 71.0],
  ],
  "Maratha Empire": [
    [24.0, 77.0], [22.0, 83.0], [18.0, 81.0], [15.0, 74.0],
    [17.0, 73.0], [20.0, 73.0], [24.0, 77.0],
  ],
};

// OpenStreetMap standard tiles: free, no API key. (CARTO's free basemaps now return
// "API key required" watermarks.) Dark mode inverts the same tiles with a CSS filter.
const TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export default function HistoricalMap({ coordinates, entityName, dynastyName, eraYears, theme }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<L.Map | null>(null);
  const [offline, setOffline] = useState(FORCE_OFFLINE);
  const [lat, lng] = coordinates;

  // create / destroy the map once
  useEffect(() => {
    if (!host.current) return;
    const m = L.map(host.current, { center: [lat, lng], zoom: 5, zoomControl: false, attributionControl: true });
    L.control.zoom({ position: "bottomright" }).addTo(m);
    setMap(m);
    const ro = new ResizeObserver(() => m.invalidateSize());
    ro.observe(host.current);
    return () => {
      ro.disconnect();
      m.remove();
      setMap(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- initial centre only
  }, []);

  // tiles (online) or vector outline (offline)
  useEffect(() => {
    if (!map) return;
    const layers: L.Layer[] = [];
    if (offline) {
      layers.push(
        L.polygon(SUBCONTINENT_OUTLINE, {
          color: theme === "dark" ? "#6b6150" : "#b9ad94",
          weight: 1.5,
          fillColor: theme === "dark" ? "#3a3428" : "#e6decb",
          fillOpacity: 0.7,
          interactive: false,
        }),
      );
    } else {
      let loaded = false;
      let errors = 0;
      const tiles = L.tileLayer(TILE_URL, { attribution: ATTRIBUTION, maxZoom: 18,
        className: theme === "dark" ? "osm-tiles-dark" : "" });
      tiles.on("tileload", () => {
        loaded = true;
      });
      tiles.on("tileerror", () => {
        errors += 1;
        if (!loaded && errors >= 2) setOffline(true);
      });
      layers.push(tiles);
    }
    layers.forEach((l) => l.addTo(map));
    return () => {
      layers.forEach((l) => l.remove());
    };
  }, [map, offline, theme]);

  // marker + boundary
  useEffect(() => {
    if (!map) return;
    const layers: L.Layer[] = [];
    const accent = "#d9731a";
    const boundary = dynastyName ? EMPIRE_BOUNDARIES[dynastyName] : undefined;
    if (boundary) {
      layers.push(
        L.polygon(boundary, { color: accent, weight: 2, fillColor: accent, fillOpacity: 0.14, dashArray: "6 6" }).addTo(map),
      );
    }
    const marker = L.circleMarker([lat, lng], {
      radius: 10,
      fillColor: accent,
      color: "#fff",
      weight: 3,
      fillOpacity: 0.9,
    }).addTo(map);
    const pop = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = entityName;
    pop.append(b);
    if (dynastyName || eraYears) {
      pop.append(document.createElement("br"), `${dynastyName ?? ""}${eraYears ? ` (${formatEra(eraYears)})` : ""}`);
    }
    marker.bindPopup(pop);
    layers.push(marker);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const sized = map.getSize().x > 0 && map.getSize().y > 0;
    if (reduce || !sized) map.setView([lat, lng], 5, { animate: false });
    else map.flyTo([lat, lng], 5, { duration: 1 });
    return () => {
      layers.forEach((l) => l.remove());
    };
  }, [map, lat, lng, entityName, dynastyName, eraYears]);

  return (
    <div className={`map-wrap${offline ? " offline" : ""}`}>
      <div ref={host} className="map" role="application" aria-label={`Map showing ${entityName}`} />
      <div className="map-badge">
        <strong>{dynastyName ?? entityName}</strong>
        <span>
          {eraYears ? `${formatEra(eraYears)} · ` : ""}
          {lat.toFixed(2)}°, {lng.toFixed(2)}°
        </span>
        {offline && <span className="chip">offline map</span>}
      </div>
    </div>
  );
}
