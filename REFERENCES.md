# Official Codebooks and Data Documentation

1. **UCDP External Support Dataset (ESD) Codebook v18.1** (Meier, 2022). Official PDF: https://ucdp.uu.se/downloads/extsup/ESD/ucdp-esd-181.pdf
   - Defines TY / AY / DY units, `dyad_id`, `actor_id`, `oppo_id`, `ext_sup`, `ext_alleged`, support type fields, same-supporter `ext_bothsides`, coalition coding and incomplete coverage of inactive conflict-years.
2. **UCDP Dyadic Dataset Codebook v18.1** (Themnér). Official PDF: https://ucdp.uu.se/downloads/dyadic/ucdp-dyadic-181.pdf
   - Defines dyad-year unit, types, intensity (`Int`: 1 Minor = 25–999 battle-related deaths; 2 War = 1,000+), historical ID versioning and missing-data convention (`-99`).
3. **UCDP Battle-Related Deaths Dataset v18.1, dyadic format**: available as a dyadic download under version 18.1 in the official historical download center. Use this unit for dyad-year robustness rather than the conflict-level file: https://ucdp.uu.se/downloads/olddw.html
4. **UCDP official historical download center and ID translation tables**: https://ucdp.uu.se/downloads/olddw.html and https://ucdp.uu.se/downloads/
   - UCDP states that IDs changed in version 17.1 and older/newer IDs do not automatically match; translation tables are required for mixed-version use.
5. **Meier, V., Karlén, N., Pettersson, T., & Croicu, M. (2023).** “External Support in Armed Conflicts: Introducing the UCDP External Support Dataset (ESD), 1975–2017.” *Journal of Peace Research*. DOI: https://doi.org/10.1177/00223433221079864
6. **COW National Material Capabilities v7.0**: https://correlatesofwar.org/data-sets/national-material-capabilities/
   - Data/codebook and documentation for CINC and six component indicators; raw and supplemental files are provided.
7. **World Development Indicators (WDI), sources and methods**: https://datatopics.worldbank.org/world-development-indicators/sources-and-methods.html
   - World Bank notes that availability, reliability and comparability vary; country coverage can be incomplete, including because of conflict-related collection/reporting problems.
8. **V-Dem Dataset v16 (March 2026) and codebook downloads**: https://www.v-dem.net/data/the-v-dem-dataset/
   - Official release includes data, codebook, release notes, cautionary notes and suggested citation.
9. **COW Formal Alliances**: https://correlatesofwar.org/data-sets/formal-alliances/
10. **COW Militarized Interstate Disputes**: https://correlatesofwar.org/data-sets/mids/

The pipeline has been written against the documented v18.1 schema but should still be run on the user's downloaded files and checked against the generated audit reports. Source releases can change field names and the exact downloadable file format.
