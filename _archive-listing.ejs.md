```{=html}
<div class="archive-list">
<%
let currentYear = null;
const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function archiveDate(rawDate) {
  const value = String(rawDate || "");
  const monthFirst = value.match(/^([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})$/);
  if (monthFirst) {
    return { year: monthFirst[3], short: `${monthFirst[2].padStart(2, "0")} ${monthFirst[1].slice(0, 3)}` };
  }

  const iso = value.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (iso) {
    return { year: iso[1], short: `${iso[3]} ${monthNames[Number(iso[2]) - 1]}` };
  }

  const parsed = new Date(rawDate);
  if (!Number.isNaN(parsed.getTime())) {
    return {
      year: String(parsed.getFullYear()),
      short: `${String(parsed.getDate()).padStart(2, "0")} ${monthNames[parsed.getMonth()]}`
    };
  }

  return { year: "Undated", short: "" };
}

for (const item of items) {
  const filenameDate = item.filename
    ? item.filename.match(/^(\d{4}-\d{2}-\d{2})/)
    : null;
  const rawDate = item.date || (filenameDate ? filenameDate[1] : null) || item["file-modified"];
  const { year, short: shortDate } = archiveDate(rawDate);

  if (year !== currentYear) {
    if (currentYear !== null) {
%>
    </div>
  </section>
<%
    }
    currentYear = year;
%>
  <section class="archive-year">
    <h2><%= year %></h2>
    <div class="archive-items">
<%
  }
%>
      <article class="archive-item" <%= metadataAttrs(item) %>>
        <time class="archive-date listing-date" datetime="<%- rawDate || "" %>"><%= shortDate %></time>
        <a class="archive-title listing-title" href="<%- item.path %>"><%- item.title %></a>
      </article>
<%
}

if (currentYear !== null) {
%>
    </div>
  </section>
<%
}
%>
</div>
```
