var pics=new Array();
var count=0;
var i=0;
var ith=0;
var speed=1000;
var data="";
var kk;
var dateStr;
                                                                                                                                                             
  function myWin(){
    newWin = open ("../../gif_js/main.htm", "displayWindow", "width=850,height=400,menubar=yes,resizable=yes,scrollbars=yes,toolbar=yes,location=yes,status=yes");
                                                                                                                                                             
  }
                                                                                                                                                             
  function getTime() {
                                                                                                                                                             
    now = new Date();
    var z;
    h   = now.getHours();
    if( h<11 ) z=1;
    else z=0;
    date = now.getDate() - z;
    month = now.getMonth() + 1;
    yearStr += now.getYear() + 1900;
    dateStr ="" + yearStr;
    dateStr += + month
    dateStr += ((date < 10) ? "0" : "") + date;
  }
                                                                                                                                                             
                                                                                                                                                             
                                                                                                                                                             
function preload(img){
     pics[count] = new Image();
     pics[count].src = img;
     count++;
}

function get_speed(frm){
  if(frm.elements["spd"].selectedIndex == 0) speed=1000;
  if(frm.elements["spd"].selectedIndex == 1) speed=100;
  if(frm.elements["spd"].selectedIndex == 2) speed=2000;
  return speed;
}
                                                                                                                                                             
                                                                                                                                                             
                                                                                                                                                             
function increse_i(){
   i++;
   return i;
}
                                                                                                                                                             
                                                                                                                                                             
                                                                                                                                                             
function nxt(){
     if(i>=22){
       i=0;
//       return;
     }
     document.my_image.src =  pics[i].src;
     window.setTimeout("increse_i(); nxt()", speed);
}
                                                                                                                                                             
                                                                                                                                                             
function show(i){
  ith=i;
  document.my_image.src=pics[ith].src;
}

function next(){
  ith=ith+1;
  if(ith >= 21) ith=21;
   document.my_image.src=pics[ith].src;
}
                                                                                                                                                             
function prev(){
  ith=ith-1;
  if(ith < 0) ith=0;
   document.my_image.src=pics[ith].src;
}
                                                                                                                                                             
function rewind(){
  ith=0;
   document.my_image.src=pics[ith].src;
}
                                                                                                                                                             
                                                                                                                                                             
function openWin(url) {
  newWin=window.open(url);
}
                                                                                                                                                             
function reload(){
   open(this);
}

function load_image(frm){
  count=0;
  for(k=1; k<13; k++){
    kk=k*3
    if(kk<10) data="0"+kk;
    else data=kk;
    filename="../../"+frm.dy.options[frm.dy.selectedIndex].value+"/"+frm.prob.options[frm.prob.selectedIndex].value+".t20z."+data+".gif";
    preload(filename);
  }
  show(0);
}

function animation(){
     if(i>=12){
       i=0;
     }
     if(document.form2.timerBox.checked){
       document.my_image.src =  pics[i].src;
       window.setTimeout("increse_i(); animation()", speed);
       ith=i;
     }
     else{
       return;
     }
}


