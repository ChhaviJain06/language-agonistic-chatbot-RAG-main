import { useState } from "react";
import { IconBook, IconGraph, IconMap } from "../lib/icons";
import { formatEra, wikiHref } from "../lib/text";
import type { DrawerItem, Source, VisualStageData } from "../lib/types";
import EraBar from "./EraBar";
import HistoricalMap from "./HistoricalMap";
import SubgraphGraph from "./SubgraphGraph";
import Sources from "./Sources";

type Tab = "map" | "graph" | "context";

interface Props {
  data: VisualStageData | null;
  sources: Source[];
  theme: "light" | "dark";
  onOpen: (s: DrawerItem) => void;
}

function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="stage-empty">
      <p className="stage-empty-title">{title}</p>
      <p>{body}</p>
    </div>
  );
}

export default function VisualStage({ data, sources, theme, onOpen }: Props) {
  const [tab, setTab] = useState<Tab>("map");
  const matched = !!data?.matched;
  const nodes = data?.subgraph?.nodes ?? [];
  const links = data?.subgraph?.links ?? [];
  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "map", label: "Map", icon: <IconMap size={15} /> },
    { id: "graph", label: "Graph", icon: <IconGraph size={15} /> },
    { id: "context", label: "Sources & timeline", icon: <IconBook size={15} /> },
  ];
  const cite = data?.citation;
  const citeHref = cite ? wikiHref(cite.url) : null;

  return (
    <div className="stage">
      <div className="stage-head">
        <div className="tabs" role="tablist" aria-label="Visual Stage">
          {tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              type="button"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              className={`tab${tab === t.id ? " active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.icon}
              <span>{t.label}</span>
            </button>
          ))}
        </div>
        {matched && data?.entity && (
          <div className="stage-entity">
            <strong>{data.entity}</strong>
            {data.dynasty && <span>{data.dynasty}</span>}
          </div>
        )}
      </div>

      <div className="stage-body" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === "map" &&
          (matched && data?.geo_coordinates ? (
            <>
              <HistoricalMap
                coordinates={data.geo_coordinates}
                entityName={data.entity ?? "Location"}
                dynastyName={data.dynasty}
                eraYears={data.era_years}
                theme={theme}
              />
              {data.era_years && <EraBar years={data.era_years} />}
            </>
          ) : (
            <Empty
              title={data ? "No map location for this answer" : "Nothing to show yet"}
              body={
                data
                  ? "The knowledge base did not match a place or dynasty for this question, so no map is drawn."
                  : "Ask a question. When the answer relates to a known entity, its location and era appear here."
              }
            />
          ))}

        {tab === "graph" &&
          (matched && nodes.length > 0 ? (
            <SubgraphGraph nodes={nodes} links={links} />
          ) : (
            <Empty
              title="No knowledge graph for this answer"
              body="Relationships appear here when the question matches entities in the knowledge graph."
            />
          ))}

        {tab === "context" && (
          <div className="context">
            {matched && data?.era_years ? (
              <EraBar years={data.era_years} />
            ) : (
              <p className="muted">No era information for this answer{data?.era_years ? ` (${formatEra(data.era_years)})` : ""}.</p>
            )}
            {cite && (
              <article className="cite-card">
                <h4>Key passage</h4>
                <div className="source-article">{cite.article}</div>
                {cite.section && <div className="source-section">{cite.section}</div>}
                <blockquote className="source-text">{cite.text}</blockquote>
                <div className="row">
                  <button type="button" className="btn ghost sm" onClick={() => onOpen(cite)}>
                    Open in drawer
                  </button>
                  {citeHref && (
                    <a className="btn ghost sm" href={citeHref} target="_blank" rel="noopener noreferrer">
                      Wikipedia
                    </a>
                  )}
                </div>
              </article>
            )}
            {sources.length > 0 ? (
              <Sources sources={sources} onOpen={onOpen} />
            ) : (
              !cite && <Empty title="No sources yet" body="Sources of the latest answer are listed here." />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
