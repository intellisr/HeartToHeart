var windowFocus = true;
var username;
var chatHeartbeatCount = 0;
var minChatHeartbeat = 1000;
var maxChatHeartbeat = 33000;
var chatHeartbeatTime = 1000;// minChatHeartbeat;
var originalTitle;
var blinkOrder = 0;
var chatboxFocus = new Array();
var newMessages = new Array();
var newMessagesWin = new Array();
var chatBoxes = new Array();


$(document).ready(function(){
	originalTitle = document.title;
	startChatSession();
  getOnlineAvalableUsers();
	$([window, document]).blur(function(){
		windowFocus = false;
	}).focus(function(){
		windowFocus = true;
		document.title = originalTitle;
	});
});
setInterval(function(){
   getOnlineAvalableUsers();
  },18000);
function getOnlineAvalableUsers(){
    var keyword = $('#search_').val();
    $.post(http_path+"chat/UpdateTime", '', function(data){	});
    if(keyword!=''){
      $('#panel-33').load(http_path+"chat/view",{'search':keyword});
    }else{
      $('#panel-33').load(http_path+"chat/view");
    }
//  
//  $.post(http_path+"chat/UpdateTime", '', function(data){	});
}

function restructureChatBoxes() {
	align = 0;
	for (x in chatBoxes) {
		chatboxtitle = chatBoxes[x];

		if ($("#chatbox_"+chatboxtitle).css('display') != 'none') {
			if (align == 0) {
				$("#chatbox_"+chatboxtitle).css('right', '20px');
			} else {
				width = (align)*(225+7)+20;
				$("#chatbox_"+chatboxtitle).css('right', width+'px');
			}
			align++;
		}
	}
}

function chatWith(chatuser,fName) {
	createChatBox(chatuser,'',fName);
	$("#chatbox_"+chatuser+" .chatboxtextarea").focus();
}

function createChatBox(chatboxtitle,minimizeChatBox,fromname) {
	if ($("#chatbox_"+chatboxtitle).length > 0) {

		if ($("#chatbox_"+chatboxtitle).css('display') == 'none') {
			$("#chatbox_"+chatboxtitle).css('display','block');
			
			restructureChatBoxes();
		}
		$("#chatbox_"+chatboxtitle+" .chatboxtextarea").focus();
		return;
	}

	$("<div />" ).attr("id","chatbox_"+chatboxtitle)
	.addClass("chatbox")
	.html('<div class="chatboxhead" id="chatheaderID_'+chatboxtitle+'"><div class="chatboxtitle">'+fromname+'</div><div class="chatboxoptions">\n\
<a href="javascript:void(0)" onclick="javascript:toggleChatBoxGrowth(\''+chatboxtitle+'\')"><img src="'+http_path+'images/smiles/chat-minimize.png"></a> \n\
<a href="javascript:void(0)" onclick="javascript:closeChatBox(\''+chatboxtitle+'\')"><img src="'+http_path+'images/smiles/chat-close.png"></a></div><br clear="all"/></div>\n\
<div class="chatboxcontent"></div><div class="chatboxinput">\n\
<textarea id="txtareaID_'+chatboxtitle+'" class="chatboxtextarea" onkeydown="javascript:return checkChatBoxInputKey(event,this,\''+chatboxtitle+'\',\''+chatOwner+'\');"></textarea>\n\
         <img id="smile-ic" src="'+http_path+'images/smiles/smile-group.png" onclick="javascript:loadSmilesPopup(\''+chatboxtitle+'\',\''+chatOwner+'\')" /><div class="smile-bar"  id="loadSmilesPopup_'+chatboxtitle+'"></div></div>')
         //<img src="'+http_path+'images/smiles/smile-group.png" onclick="javascript:loadSmilesPopup(\''+chatboxtitle+'\',\''+chatOwner+'\')" /><img src="'+http_path+'images/smiles/smile-group_2.png" onclick="javascript:loadSmilesPopup2(\''+chatboxtitle+'\',\''+chatOwner+'\')" /><img src="'+http_path+'images/smiles/bigsmile.png" onclick="javascript:loadSmilesPopup3(\''+chatboxtitle+'\',\''+chatOwner+'\')" /><div class="smile-bar"  id="loadSmilesPopup_'+chatboxtitle+'"></div></div>')
	.appendTo($( "body" ));
		
	$("#chatbox_"+chatboxtitle).css('bottom', '0px');
	
	chatBoxeslength = 0;

	for (x in chatBoxes) {
		if ($("#chatbox_"+chatBoxes[x]).css('display') != 'none') {
			chatBoxeslength++;
		}
	}

	if (chatBoxeslength == 0) {
		$("#chatbox_"+chatboxtitle).css('right', '20px');
	} else {
		width = (chatBoxeslength)*(225+7)+20;
		$("#chatbox_"+chatboxtitle).css('right', width+'px');
	}
	
	chatBoxes.push(chatboxtitle);

	if (minimizeChatBox == 1) {
		minimizedChatBoxes = new Array();

		if ($.cookie('chatbox_minimized')) {
			minimizedChatBoxes = $.cookie('chatbox_minimized').split(/\|/);
		}
		minimize = 0;
		for (j=0;j<minimizedChatBoxes.length;j++) {
			if (minimizedChatBoxes[j] == chatboxtitle) {
				minimize = 1;
			}
		}

		if (minimize == 1) {
			$('#chatbox_'+chatboxtitle+' .chatboxcontent').css('display','none');
			$('#chatbox_'+chatboxtitle+' .chatboxinput').css('display','none');
		}
	}

	chatboxFocus[chatboxtitle] = false;

	$("#chatbox_"+chatboxtitle+" .chatboxtextarea").blur(function(){
		chatboxFocus[chatboxtitle] = false;
		$("#chatbox_"+chatboxtitle+" .chatboxtextarea").removeClass('chatboxtextareaselected');
	}).focus(function(){
		chatboxFocus[chatboxtitle] = true;
		newMessages[chatboxtitle] = false;
		$('#chatbox_'+chatboxtitle+' .chatboxhead').removeClass('chatboxblink');
		$("#chatbox_"+chatboxtitle+" .chatboxtextarea").addClass('chatboxtextareaselected');
	});

	$("#chatbox_"+chatboxtitle).click(function() {
		if ($('#chatbox_'+chatboxtitle+' .chatboxcontent').css('display') != 'none') {
			$("#chatbox_"+chatboxtitle+" .chatboxtextarea").focus();
		}
	});

	$("#chatbox_"+chatboxtitle).show();
}
function loadSmilesPopup(chatboxtitle,fromname){
    $.post(http_path+"/chat/loadSmilesPopup", {chatboxtitle: chatboxtitle,fromname: fromname} , function(data){ 
        $("#loadSmilesPopup_"+chatboxtitle).html(data); 
	});
}
function loadSmilesPopup2(chatboxtitle,fromname){
    $.post(http_path+"/chat/loadSmilesPopup2", {chatboxtitle: chatboxtitle,fromname: fromname} , function(data){ 
        $("#loadSmilesPopup_"+chatboxtitle).html(data); 
	});
}
function loadSmilesPopup3(chatboxtitle,fromname){
    $.post(http_path+"/chat/loadSmilesPopup3", {chatboxtitle: chatboxtitle,fromname: fromname} , function(data){ 
        $("#loadSmilesPopup_"+chatboxtitle).html(data); 
	});
}
function setSmiles(message,chatboxtitle,fromname){

		message = message.replace(/^\s+|\s+$/g,"");
                $("#loadSmilesPopup_"+chatboxtitle).html("");  
		if (message != '') {
			$.post(http_path+"chat/sendChat", {to: chatboxtitle, message: message,frmn:fromname} , function(data){
				var str = data.split('sep1sep');				
					
				if(str[1]=="1")		
				{
					$('#chatheaderID_'+chatboxtitle).removeClass().addClass('chatboxhead_new');	
					$('#txtareaID_'+chatboxtitle).attr("disabled","disabled"); 
				}

					
				message = message.replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\"/g,"&quot;");
				$("#chatbox_"+chatboxtitle+" .chatboxcontent").append('<div class="chatboxmessage"><span class="chatboxmessagefrom">'+fromname+':&nbsp;&nbsp;</span><span class="chatboxmessagecontent">'+str[0]+'</span></div>');
				$("#chatbox_"+chatboxtitle+" .chatboxcontent").scrollTop($("#chatbox_"+chatboxtitle+" .chatboxcontent")[0].scrollHeight);
			});
		}
		chatHeartbeatTime = minChatHeartbeat;
		chatHeartbeatCount = 1;
                
           
		return false;
    
    
    
    
    
    

 
}
function chatHeartbeat(){
	//alert('hi');
	var itemsfound = 0;
	
	if (windowFocus == false) {
 
		var blinkNumber = 0;
		var titleChanged = 0;
                var titlName = "";
		for (x in newMessagesWin) {
			if (newMessagesWin[x] == true) {
				++blinkNumber;
				if (blinkNumber >= blinkOrder) {
					 $.post(http_path+"/chat/getChatUserName", { id: x} , function(data){
                                        titlName = data;
	                          });
                                    
                                    document.title = titlName+' says...';
					titleChanged = 1;
					break;	
				}
			}
		}
		
		if (titleChanged == 0) {
			document.title = originalTitle;
			blinkOrder = 0;
		} else {
			++blinkOrder;
		}

	} else {
		for (x in newMessagesWin) {
			newMessagesWin[x] = false;
		}
	}

	for (x in newMessages) {
		if (newMessages[x] == true) {
			if (chatboxFocus[x] == false) {
				//FIXME: add toggle all or none policy, otherwise it looks funny
				$('#chatbox_'+x+' .chatboxhead').toggleClass('chatboxblink');
			}
		}
	}
		
	$.ajax({
	  url: http_path+"chat/chatHeartBeat",
	  cache: false,
	  dataType: "json",
	  success: function(data) {
		
		$.each(data.items, function(i,item){
			if (item)	{ // fix strange ie bug
				chatboxtitle = item.f;
				var fromname = item.fnm;
				if ($("#chatbox_"+chatboxtitle).length <= 0) {
					createChatBox(chatboxtitle,'',fromname);
				}
				if ($("#chatbox_"+chatboxtitle).css('display') == 'none') {
					$("#chatbox_"+chatboxtitle).css('display','block');
					restructureChatBoxes();
				}
				
				if (item.s == 1) {
					item.f = username;
				}

				if (item.s == 2) {
					$("#chatbox_"+chatboxtitle+" .chatboxcontent").append('<div class="chatboxmessage"><span class="chatboxinfo">'+item.m+'</span></div>');
				} else {
					newMessages[chatboxtitle] = true;
					newMessagesWin[chatboxtitle] = true;
					$("#chatbox_"+chatboxtitle+" .chatboxcontent").append('<div class="chatboxmessage"><span class="chatboxmessagefrom">'+item.fn+':&nbsp;&nbsp;</span><span class="chatboxmessagecontent">'+item.m+'</span></div>');
				}

				$("#chatbox_"+chatboxtitle+" .chatboxcontent").scrollTop($("#chatbox_"+chatboxtitle+" .chatboxcontent")[0].scrollHeight);
				itemsfound += 1;
			}
		});

		chatHeartbeatCount++;

		if (itemsfound > 0) {
			chatHeartbeatTime = minChatHeartbeat;
			chatHeartbeatCount = 1;
		} else if (chatHeartbeatCount >= 10) {
			chatHeartbeatTime *= 2;
			chatHeartbeatCount = 1;
			if (chatHeartbeatTime > maxChatHeartbeat) {
				chatHeartbeatTime = maxChatHeartbeat;
			}
		}
		setTimeout('chatHeartbeat();',chatHeartbeatTime);
	}});
}

function closeChatBox(chatboxtitle) {
	
	$('#chatbox_'+chatboxtitle).css('display','none');
	restructureChatBoxes();

	$.post(http_path+"/chat/closechat", { chatbox: chatboxtitle} , function(data){	
	});

}

function toggleChatBoxGrowth(chatboxtitle) {
	if ($('#chatbox_'+chatboxtitle+' .chatboxcontent').css('display') == 'none') {  
		
		var minimizedChatBoxes = new Array();
		
		if ($.cookie('chatbox_minimized')) {
			minimizedChatBoxes = $.cookie('chatbox_minimized').split(/\|/);
		}
		var newCookie = '';

		for (i=0;i<minimizedChatBoxes.length;i++) {
			if (minimizedChatBoxes[i] != chatboxtitle) {
				newCookie += chatboxtitle+'|';
			}
		}
		newCookie = newCookie.slice(0, -1)

		$.cookie('chatbox_minimized', newCookie);
		$('#chatbox_'+chatboxtitle+' .chatboxcontent').css('display','block');
		$('#chatbox_'+chatboxtitle+' .chatboxinput').css('display','block');
		$("#chatbox_"+chatboxtitle+" .chatboxcontent").scrollTop($("#chatbox_"+chatboxtitle+" .chatboxcontent")[0].scrollHeight);
	} else {
		
		var newCookie = chatboxtitle;

		if ($.cookie('chatbox_minimized')) {
			newCookie += '|'+$.cookie('chatbox_minimized');
		}

		$.cookie('chatbox_minimized',newCookie);
		$('#chatbox_'+chatboxtitle+' .chatboxcontent').css('display','none');
		$('#chatbox_'+chatboxtitle+' .chatboxinput').css('display','none');
	}
	
}

function checkChatBoxInputKey(event,chatboxtextarea,chatboxtitle,fromname) {
	//alert(fromname);
	if(event.keyCode == 13 && event.shiftKey == 0)  {
		message = $(chatboxtextarea).val();
		message = message.replace(/^\s+|\s+$/g,"");
		$(chatboxtextarea).val('');
		$(chatboxtextarea).focus();
		$(chatboxtextarea).css('height','44px');
		if (message != '') {
			$.post(http_path+"chat/sendChat", {to: chatboxtitle, message: message,frmn:fromname} , function(data){
				var str = data.split('sep1sep');				
					
				if(str[1]=="1")		
				{
					$('#chatheaderID_'+chatboxtitle).removeClass().addClass('chatboxhead_new');	
					$('#txtareaID_'+chatboxtitle).attr("disabled","disabled"); 
				}

					
				message = message.replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\"/g,"&quot;");
				$("#chatbox_"+chatboxtitle+" .chatboxcontent").append('<div class="chatboxmessage"><span class="chatboxmessagefrom">'+fromname+':&nbsp;&nbsp;</span><span class="chatboxmessagecontent">'+str[0]+'</span></div>');
				$("#chatbox_"+chatboxtitle+" .chatboxcontent").scrollTop($("#chatbox_"+chatboxtitle+" .chatboxcontent")[0].scrollHeight);
			});
		}
		chatHeartbeatTime = minChatHeartbeat;
		chatHeartbeatCount = 1;

		return false;
	}

	var adjustedHeight = chatboxtextarea.clientHeight;
	var maxHeight = 94;

	if (maxHeight > adjustedHeight) {
		adjustedHeight = Math.max(chatboxtextarea.scrollHeight, adjustedHeight);
		if (maxHeight)
			adjustedHeight = Math.min(maxHeight, adjustedHeight);
		if (adjustedHeight > chatboxtextarea.clientHeight)
			$(chatboxtextarea).css('height',adjustedHeight+8 +'px');
	} else {
		$(chatboxtextarea).css('overflow','auto');
	}

}

function startChatSession(){  

	$.ajax({
	  url: http_path+"chat/startChatSession",
	  cache: false,
	  dataType: "json",
	  success: function(data) {
         	//alert(data.logD);
		//username = data.username;
		username = data.username;
		var login = data.logD;
		$.each(data.items, function(i,item){
			if (item)	{ // fix strange ie bug
				chatboxtitle = item.f;
				var fromname = item.fn;
				var fromnewname = item.fnn;
				
				var newfrmname = "";
				//alert(login);				
/*				if(login=="LO"){
					$('#chatheaderID_'+chatboxtitle).removeClass().addClass('chatboxhead_new');	
					$('#txtareaID_'+chatboxtitle).attr("disabled","disabled"); 
				}*/
					
				if(fromnewname=="no")
					newfrmname=fromname;
				else
					newfrmname = fromnewname;

				if ($("#chatbox_"+chatboxtitle).length <= 0) {
					createChatBox(chatboxtitle,1,fromname);
				}
				
				if (item.s == 1) {
					item.f = username;
				}

				if (item.s == 2) {
					$("#chatbox_"+chatboxtitle+" .chatboxcontent").append('<div class="chatboxmessage"><span class="chatboxinfo">'+item.m+'</span></div>');
				} else {
					$("#chatbox_"+chatboxtitle+" .chatboxcontent").append('<div class="chatboxmessage"><span class="chatboxmessagefrom">'+newfrmname+':&nbsp;&nbsp;</span><span class="chatboxmessagecontent">'+item.m+'</span></div>');
				}
			}
		});
		
		for (i=0;i<chatBoxes.length;i++) {
			chatboxtitle = chatBoxes[i];
			$("#chatbox_"+chatboxtitle+".chatboxcontent").scrollTop($("#chatbox_"+chatboxtitle+" .chatboxcontent")[0].scrollHeight);
			setTimeout('$("#chatbox_"+chatboxtitle+" .chatboxcontent").scrollTop($("#chatbox_"+chatboxtitle+" .chatboxcontent")[0].scrollHeight);', 100); // yet another strange ie bug
		}

	setTimeout('chatHeartbeat();',chatHeartbeatTime);
		
	}});
}

/**
 * Cookie plugin
 *
 * Copyright (c) 2006 Klaus Hartl (stilbuero.de)
 * Dual licensed under the MIT and GPL licenses:
 * http://www.opensource.org/licenses/mit-license.php
 * http://www.gnu.org/licenses/gpl.html
 *
 */

jQuery.cookie = function(name, value, options) {
    if (typeof value != 'undefined') { // name and value given, set cookie
        options = options || {};
        if (value === null) {
            value = '';
            options.expires = -1;
        }
        var expires = '';
        if (options.expires && (typeof options.expires == 'number' || options.expires.toUTCString)) {
            var date;
            if (typeof options.expires == 'number') {
                date = new Date();
                date.setTime(date.getTime() + (options.expires * 24 * 60 * 60 * 1000));
            } else {
                date = options.expires;
            }
            expires = '; expires=' + date.toUTCString(); // use expires attribute, max-age is not supported by IE
        }
        // CAUTION: Needed to parenthesize options.path and options.domain
        // in the following expressions, otherwise they evaluate to undefined
        // in the packed version for some reason...
        var path = options.path ? '; path=' + (options.path) : '';
        var domain = options.domain ? '; domain=' + (options.domain) : '';
        var secure = options.secure ? '; secure' : '';
        document.cookie = [name, '=', encodeURIComponent(value), expires, path, domain, secure].join('');
    } else { // only name given, get cookie
        var cookieValue = null;
        if (document.cookie && document.cookie != '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) == (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
};