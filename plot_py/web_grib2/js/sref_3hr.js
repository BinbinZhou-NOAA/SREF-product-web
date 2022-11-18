var pics=new Array();
var count=0;
var i=0;
var ith=0;
var speed=1000;
var data="";
var kk;
var dateStr;
var filename="";
var begin=1;

  function myWin(){
    newWin = open ("../js/main.htm", "displayWindow", "width=850,height=400,menubar=yes,resizable=yes,scrollbars=yes,toolbar=yes,location=yes,status=yes");
  }
  

//  function myWin(rr){
//    newWin = open ("../../back_see/html/SBACK.COM_US"+".html", "displayWindow", "width=1000,height=800,menubar=yes,resizable=yes,scrollbars=yes,toolbar=yes,location=yes,status=yes");
//  }
                                                                                                                                                           
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
                                                                                                                                                             
           
function animation(){
     if(i>=13-begin){
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

function get_i(){
 i=ith;
}

                                                                                                                                                  
function show(i){
  ith=i;
  document.my_image.src=pics[ith].src;
}

function next(){
  ith=ith+1;
  if(ith >= 12-begin) ith=12-begin;;
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
                                                                                                                                                             
function last(){
  ith=12-begin;
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
  if(frm.elements["type"].selectedIndex == 4 ) begin=1;
  for(k=begin; k<13; k++){
    kk=k*3
    if(kk<10) data="0"+kk;
    else data=kk;
    filename="../../"+frm.dy.options[frm.dy.selectedIndex].value+"/"+frm.cyc.options[frm.cyc.selectedIndex].value+"/"+ frm.reg.options[frm.reg.selectedIndex].value+"/"+frm.sub.options[frm.sub.selectedIndex].value+"/"+frm.prod.options[frm.prod.selectedIndex].value+".t"+frm.cyc.options[frm.cyc.selectedIndex].value+"z."+data+".gif";
    preload(filename);
  }
  show(0);
}


