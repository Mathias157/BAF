*Industry add-on. Implemented by Juan Gea Bermudez 2019

$onmulti
$if     EXIST '../data/INDUSTRY_AAA.inc' $INCLUDE         '../data/INDUSTRY_AAA.inc';
$if not EXIST '../data/INDUSTRY_AAA.inc' $INCLUDE '../../base/data/INDUSTRY_AAA.inc';
$offmulti

$onmulti
$if     EXIST '../data/INDUSTRY_INDUSTRY_AAA.inc' $INCLUDE         '../data/INDUSTRY_INDUSTRY_AAA.inc';
$if not EXIST '../data/INDUSTRY_INDUSTRY_AAA.inc' $INCLUDE '../../base/data/INDUSTRY_INDUSTRY_AAA.inc';
$offmulti
