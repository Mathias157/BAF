* Latest revision 20171023(hr)


$ifi %testbb1234%==yes    $include "../../base/addons/bb1234compare/bb1234_vardeclare.inc";
$ifi %HYRSBB123%==quantprice      $include "../../base/addons/hyrsbb123/hyrsbb123variables.inc";
$ifi %dflexquant%==yes            $include "../../base/addons/demandflexquant/dflexquant_vardeclare.inc";

$ifi %BB4%==yes $goto beginbb4
$ifi %FLOWBASED%==yes  $include   "../../base/addons/flowbased/flowbased_vardeclare.inc";
$ifi %HYDROGEN%==yes  $include   "../../base/addons/hydrogen/hydrogen_vardeclare.inc";

$ifi not %BB4%==yes $goto endoffile

$label  beginbb4

$ifi not %bb4%==yes $goto endoffile
$ifi %HYDROGEN%==yes      $include   "../../base/addons/hydrogen/bb4/hydrogen_vardeclare.inc";
$ifi %OFFSHOREGRID%==yes  $include   "../../base/addons/offshoregrid/bb4/offshoregrid_vardeclare.inc";
$ifi %TRANSPORT%==yes       $include   "../../base/addons/transport/bb4/transport_vardeclare.inc";
$ifi %UnitComm%==yes  $include   "../../base/addons/unitcommitment/bb4/uc_vardeclare.inc";
$ifi %STEPWISE_RAMPING%==yes  $include   "../../base/addons/STEPWISE_RAMPING/stepwise_ramping_vardeclare.inc";
$ifi %AGKNDISC%==yes      $include   "../../base/addons/AGKNdisc/bb4/agkndisc_vardeclare.inc";
$ifi %STEPWISEPRICE%==yes  $include   "../../base/addons/stepwiseprice/stepwiseprice_vardeclare.inc";
$ifi %X3V%==yes  $include   "../../base/addons/x3v/bb4/x3v_vardeclare.inc";
$ifi %LIMITSFORFUELS%==yes  $include   "../../base/addons/limitsforfuels/bb4/limitsforfuels_vardeclare.inc";
$ifi %GRIDTARIFFS%==yes  $include   "../../base/addons/GridTariffs/bb4/GridTariffs_vardeclare.inc";
$ifi %DEMANDRESPONSE%==yes      $include        "../../base/addons/demandresponse/bb4/dr_vardeclare.inc";
$ifi %HEATTRANS%==yes       $include   "../../base/addons/heattrans/bb4/heattrans_vardeclare.inc";
$ifi %DECOM%==yes  $include   "../../base/addons/DECOM/DECOM_vardeclare.inc";
$ifi %EV%==yes      $include        "../../base/addons/EV/bb4/EV_vardeclare.inc";
$ifi %ANC_SERV%==yes  $include   "../../base/addons/ancilliary_services/ancilliary_services_vardeclare.inc";
$ifi %FlowBased%==yes      $include        "../../base/addons/FlowBased/bb4/flowbased_vardeclare.inc";
$ifi %INDIVUSERS%==yes       $include   "../../base/addons/indivusers/bb4/indivusers_vardeclare.inc";
$ifi %BalancingMarket%==yes       $include   "../../base/addons/balancing_market/balancing_market_vardeclare.inc";
$ifi %OPTIFLOW%==yes      $include   "../../base/addons/optiflow/bb4/optiflow_vardeclare.inc";
$ifi %FLEXDEM%==yes      $include   "../../base/addons/flexdem/bb4/flexdem_vardeclare.inc";
$ifi %import_results%==yes  $include   "../../base/addons/import_results/import_results_vardeclare.inc";
$ifi %ANTBALM%==yes      $include   "../../base/addons/antbalm/bb4/antbalm_vardeclare.inc";

$ifi %vqdebug%==yes       $include   "../../base/addons/vqdebug/bb4/vqdebug_declare.inc";



$label endoffile
