﻿/** EasyWeb iframe v3.1.5 date:2019-10-05 License By http://easyweb.vip */
layui.define(["layer"], function (f) {
    var h = layui.jquery;
    var j = layui.layer;
    var a = ".layui-layout-admin>.layui-body";
    var k = a + ">.layui-tab";
    var e = ".layui-layout-admin>.layui-side>.layui-side-scroll";
    var i = ".layui-layout-admin>.layui-header";
    var b = "admin-pagetabs";
    var d = "admin-side-nav";
    var c = "theme-admin";
    var m = {
        version: "314",
        defaultTheme: "theme-admin",
        tableName: "easyweb",
        flexible: function (n) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.flexible(n);
                    return
                }
            }
            var o = h(".layui-layout-admin").hasClass("admin-nav-mini");
            (n == undefined) && (n = o);
            if (o == n) {
                if (n) {
                    m.hideTableScrollBar();
                    h(".layui-layout-admin").removeClass("admin-nav-mini")
                } else {
                    h(".layui-layout-admin").addClass("admin-nav-mini")
                }
            }
        },
        activeNav: function (o) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.activeNav(o);
                    return
                }
            }
            if (!o) {
                o = window.location.pathname;
                o = o.substring(o.indexOf("/"))
            }
            if (o && o != "") {
                h(e + ">.layui-nav .layui-nav-item .layui-nav-child dd.layui-this").removeClass("layui-this");
                h(e + ">.layui-nav .layui-nav-item.layui-this").removeClass("layui-this");
                var r = h(e + '>.layui-nav a[lay-href="' + o + '"]');
                if (r && r.length > 0) {
                    var q = h(".layui-layout-admin").hasClass("admin-nav-mini");
                    if (h(e + ">.layui-nav").attr("lay-accordion") == "true") {
                        var n = r.parent("dd").parents(".layui-nav-child");
                        if (q) {
                            h(e + ">.layui-nav .layui-nav-itemed>.layui-nav-child").not(n).css("display", "none")
                        } else {
                            h(e + ">.layui-nav .layui-nav-itemed>.layui-nav-child").not(n).slideUp("fast")
                        }
                        h(e + ">.layui-nav .layui-nav-itemed").not(n.parent()).removeClass("layui-nav-itemed")
                    }
                    r.parent().addClass("layui-this");
                    var s = r.parent("dd").parents(".layui-nav-child").parent();
                    if (q) {
                        s.not(".layui-nav-itemed").children(".layui-nav-child").css("display", "block")
                    } else {
                        s.not(".layui-nav-itemed").children(".layui-nav-child").slideDown("fast", function () {
                            var t = r.offset().top + r.outerHeight() + 30 - m.getPageHeight();
                            var u = 50 + 65 - r.offset().top;
                            if (t > 0) {
                                h(e).animate({
                                    "scrollTop": h(e).scrollTop() + t
                                }, 100)
                            } else {
                                if (u > 0) {
                                    h(e).animate({
                                        "scrollTop": h(e).scrollTop() - u
                                    }, 100)
                                }
                            }
                        })
                    }
                    s.addClass("layui-nav-itemed");
                    h('ul[lay-filter="' + d + '"]').addClass("layui-hide");
                    var p = r.parents(".layui-nav");
                    p.removeClass("layui-hide");
                    h(i + ">.layui-nav>.layui-nav-item").removeClass("layui-this");
                    h(i + '>.layui-nav>.layui-nav-item>a[nav-bind="' + p.attr("nav-id") + '"]').parent().addClass("layui-this")
                } else {
                }
            } else {
                console.warn("active url is null")
            }
        },
        popupRight: function (n) {
            if (n.title == undefined) {
                n.title = false;
                n.closeBtn = false
            }
            if (n.fixed == undefined) {
                n.fixed = true
            }
            n.anim = -1;
            n.offset = "r";
            n.shadeClose = true;
            n.area || (n.area = "336px");
            n.skin || (n.skin = "layui-anim layui-anim-rl layui-layer-adminRight");
            n.move = false;
            return m.open(n)
        },
        open: function (q) {
            if (!q.area) {
                q.area = (q.type == 2) ? ["360px", "300px"] : "360px"
            }
            if (!q.skin) {
                q.skin = "layui-layer-admin"
            }
            if (!q.offset) {
                if (m.getPageWidth() < 768) {
                    q.offset = "15px"
                } else {
                    if (window == top) {
                        q.offset = "70px"
                    } else {
                        q.offset = "40px"
                    }
                }
            }
            if (q.fixed == undefined) {
                q.fixed = false
            }
            q.resize = q.resize != undefined ? q.resize : false;
            q.shade = q.shade != undefined ? q.shade : 0.1;
            var o = q.end;
            q.end = function () {
                j.closeAll("tips");
                o && o()
            };
            if (q.url) {
                (q.type == undefined) && (q.type = 1);
                var p = q.success;
                q.success = function (r, s) {
                    m.showLoading(r, 2);
                    h(r).children(".layui-layer-content").load(q.url, function () {
                        p ? p(r, s) : "";
                        m.removeLoading(r, false)
                    })
                }
            }
            var n = j.open(q);
            (q.data) && (m.layerData["d" + n] = q.data);
            return n
        },
        layerData: {},
        getLayerData: function (n, o) {
            if (n == undefined) {
                n = parent.layer.getFrameIndex(window.name);
                return parent.layui.admin.getLayerData(n, o)
            } else {
                if (n.toString().indexOf("#") == 0) {
                    n = h(n).parents(".layui-layer").attr("id").substring(11)
                }
            }
            var p = m.layerData["d" + n];
            if (o) {
                return p ? p[o] : p
            }
            return p
        },
        putLayerData: function (o, q, n) {
            if (n == undefined) {
                n = parent.layer.getFrameIndex(window.name);
                return parent.layui.admin.putLayerData(o, q, n)
            } else {
                if (n.toString().indexOf("#") == 0) {
                    n = h(n).parents(".layui-layer").attr("id").substring(11)
                }
            }
            var p = m.getLayerData(n);
            p || (p = {});
            p[o] = q;
            m.layerData["d" + n] = p
        },
        req: function (n, o, p, q) {
            m.ajax({
                url: n,
                data: o,
                type: q,
                dataType: "json",
                success: p
            })
        },
        ajax: function (p) {
            var o = p.header;
            p.dataType || (p.dataType = "json");
            var n = p.success;
            p.success = function (q, r, t) {
                var s;
                if ("json" == p.dataType.toLowerCase()) {
                    s = q
                } else {
                    s = m.parseJSON(q)
                }
                s && (s = q);
                if (m.ajaxSuccessBefore(s, p.url) == false) {
                    return
                }
                n(q, r, t)
            };
            p.error = function (q) {
                p.success({
                    code: q.status,
                    msg: q.statusText
                })
            };
            p.beforeSend = function (t) {
                var s = m.getAjaxHeaders(p.url);
                for (var q = 0; q < s.length; q++) {
                    t.setRequestHeader(s[q].name, s[q].value)
                }
                if (o) {
                    for (var r in o) {
                        t.setRequestHeader(r, o[r])
                    }
                }
            };
            h.ajax(p)
        },
        parseJSON: function (p) {
            if (typeof p == "string") {
                try {
                    var o = JSON.parse(p);
                    if (typeof o == "object" && o) {
                        return o
                    }
                } catch (n) {
                }
            }
        },
        showLoading: function (r, q, o) {
            var p;
            if (r != undefined && (typeof r != "string") && !(r instanceof h)) {
                q = r.type;
                o = r.opacity;
                p = r.size;
                r = r.elem
            }
            (!r) && (r = "body");
            (q == undefined) && (q = 1);
            (p == undefined) && (p = "sm");
            p = " " + p;
            var n = ['<div class="ball-loader' + p + '"><span></span><span></span><span></span><span></span></div>', '<div class="rubik-loader' + p + '"></div>', '<div class="signal-loader' + p + '"><span></span><span></span><span></span><span></span></div>'];
            h(r).addClass("page-no-scroll");
            var s = h(r).children(".page-loading");
            if (s.length <= 0) {
                h(r).append('<div class="page-loading">' + n[q - 1] + "</div>");
                s = h(r).children(".page-loading")
            }
            o && s.css("background-color", "rgba(255,255,255," + o + ")");
            s.show()
        },
        removeLoading: function (o, q, n) {
            if (!o) {
                o = "body"
            }
            if (q == undefined) {
                q = true
            }
            var p = h(o).children(".page-loading");
            if (n) {
                p.remove()
            } else {
                q ? p.fadeOut() : p.hide()
            }
            h(o).removeClass("page-no-scroll")
        },
        putTempData: function (o, p) {
            var n = m.tableName + "_tempData";
            if (p != undefined && p != null) {
                layui.sessionData(n, {
                    key: o,
                    value: p
                })
            } else {
                layui.sessionData(n, {
                    key: o,
                    remove: true
                })
            }
        },
        getTempData: function (o) {
            var n = m.tableName + "_tempData";
            var p = layui.sessionData(n);
            if (p) {
                return p[o]
            } else {
                return false
            }
        },
        rollPage: function (q) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.rollPage(q);
                    return
                }
            }
            var o = h(k + ">.layui-tab-title");
            var p = o.scrollLeft();
            if ("left" === q) {
                o.animate({
                    "scrollLeft": p - 120
                }, 100)
            } else {
                if ("auto" === q) {
                    var n = 0;
                    o.children("li").each(function () {
                        if (h(this).hasClass("layui-this")) {
                            return false
                        } else {
                            n += h(this).outerWidth()
                        }
                    });
                    o.animate({
                        "scrollLeft": n - 120
                    }, 100)
                } else {
                    o.animate({
                        "scrollLeft": p + 120
                    }, 100)
                }
            }
        },
        refresh: function (n) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.refresh(n);
                    return
                }
            }
            var p;
            if (!n) {
                p = h(k + ">.layui-tab-content>.layui-tab-item.layui-show>.admin-iframe");
                if (!p || p.length <= 0) {
                    p = h(a + ">div>.admin-iframe")
                }
            } else {
                p = h(k + '>.layui-tab-content>.layui-tab-item>.admin-iframe[lay-id="' + n + '"]');
                if (!p || p.length <= 0) {
                    p = h(a + ">.admin-iframe")
                }
            }
            if (p && p[0]) {
                try {
                    p[0].contentWindow.location.reload(true)
                } catch (o) {
                    p.attr("src", p.attr("src"))
                }
            } else {
                console.warn(n + " is not found")
            }
        },
        closeThisTabs: function (n) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.closeThisTabs(n);
                    return
                }
            }
            m.closeTabOperNav();
            var o = h(k + ">.layui-tab-title");
            if (!n) {
                if (o.find("li").first().hasClass("layui-this")) {
                    j.msg("主页不能关闭", {
                        icon: 2
                    });
                    return
                }
                o.find("li.layui-this").find(".layui-tab-close").trigger("click")
            } else {
                if (n == o.find("li").first().attr("lay-id")) {
                    j.msg("主页不能关闭", {
                        icon: 2
                    });
                    return
                }
                o.find('li[lay-id="' + n + '"]').find(".layui-tab-close").trigger("click")
            }
        },
        closeOtherTabs: function (n) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.closeOtherTabs(n);
                    return
                }
            }
            if (!n) {
                h(k + ">.layui-tab-title li:gt(0):not(.layui-this)").find(".layui-tab-close").trigger("click")
            } else {
                h(k + ">.layui-tab-title li:gt(0)").each(function () {
                    if (n != h(this).attr("lay-id")) {
                        h(this).find(".layui-tab-close").trigger("click")
                    }
                })
            }
            m.closeTabOperNav()
        },
        closeAllTabs: function () {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.closeAllTabs();
                    return
                }
            }
            h(k + ">.layui-tab-title li:gt(0)").find(".layui-tab-close").trigger("click");
            h(k + ">.layui-tab-title li:eq(0)").trigger("click");
            m.closeTabOperNav()
        },
        closeTabOperNav: function () {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.admin) {
                    top.layui.admin.closeTabOperNav();
                    return
                }
            }
            h(".layui-icon-down .layui-nav .layui-nav-child").removeClass("layui-show")
        },
        changeTheme: function (t) {
            if (t) {
                layui.data(m.tableName, {
                    key: "theme",
                    value: t
                });
                if (c == t) {
                    t = undefined
                }
            } else {
                layui.data(m.tableName, {
                    key: "theme",
                    remove: true
                })
            }
            try {
                m.removeTheme(top);
                (t && top.layui) && top.layui.link(m.getThemeDir() + t + m.getCssSuffix(), t);
                var u = top.window.frames;
                for (var p = 0; p < u.length; p++) {
                    try {
                        var r = u[p];
                        m.removeTheme(r);
                        if (t && r.layui) {
                            r.layui.link(m.getThemeDir() + t + m.getCssSuffix(), t)
                        }
                        var q = r.frames;
                        for (var o = 0; o < q.length; o++) {
                            try {
                                var n = q[o];
                                m.removeTheme(n);
                                if (t && n.layui) {
                                    n.layui.link(m.getThemeDir() + t + m.getCssSuffix(), t)
                                }
                            } catch (s) {
                            }
                        }
                    } catch (s) {
                    }
                }
            } catch (s) {
            }
        },
        removeTheme: function (n) {
            if (!n) {
                n = window
            }
            if (n.layui) {
                var o = "layuicss-theme";
                n.layui.jquery('link[id^="' + o + '"]').remove()
            }
        },
        getThemeDir: function () {
            return layui.cache.base + "theme/"
        },
        closeThisDialog: function () {
            parent.layer.close(parent.layer.getFrameIndex(window.name))
        },
        closeDialog: function (n) {
            var o = h(n).parents(".layui-layer").attr("id").substring(11);
            j.close(o)
        },
        iframeAuto: function () {
            parent.layer.iframeAuto(parent.layer.getFrameIndex(window.name))
        },
        getPageHeight: function () {
            return document.documentElement.clientHeight || document.body.clientHeight
        },
        getPageWidth: function () {
            return document.documentElement.clientWidth || document.body.clientWidth
        },
        getCssSuffix: function () {
            var n = ".css";
            if (m.version != undefined) {
                n += "?v=";
                if (m.version == true) {
                    n += new Date().getTime()
                } else {
                    n += m.version
                }
            }
            return n
        },
        hideTableScrollBar: function (p) {
            if (m.getPageWidth() > 768) {
                if (!p) {
                    var o = h(k + ">.layui-tab-content>.layui-tab-item.layui-show>.admin-iframe");
                    if (o.length <= 0) {
                        o = h(a + ">div>.admin-iframe")
                    }
                    if (o.length > 0) {
                        p = o[0].contentWindow
                    }
                }
                try {
                    if (p && p.layui && p.layui.jquery) {
                        if (window.hsbTimer) {
                            clearTimeout(hsbTimer)
                        }
                        p.layui.jquery(".layui-table-body.layui-table-main").addClass("no-scrollbar");
                        window.hsbTimer = setTimeout(function () {
                            if (p && p.layui && p.layui.jquery) {
                                p.layui.jquery(".layui-table-body.layui-table-main").removeClass("no-scrollbar")
                            }
                        }, 500)
                    }
                } catch (n) {
                }
            }
        },
        modelForm: function (o, r, n) {
            var q = h(o);
            q.addClass("layui-form");
            if (n) {
                q.attr("lay-filter", n)
            }
            var p = q.find(".layui-layer-btn .layui-layer-btn0");
            p.attr("lay-submit", "");
            p.attr("lay-filter", r)
        },
        btnLoading: function (o, p, q) {
            if (p != undefined && (typeof p == "boolean")) {
                q = p;
                p = undefined
            }
            (q == undefined) && (q = true);
            var n = h(o);
            if (q) {
                p && n.html(p);
                n.find(".layui-icon").addClass("layui-hide");
                n.addClass("icon-btn");
                n.prepend('<i class="layui-icon layui-icon-loading layui-anim layui-anim-rotate layui-anim-loop ew-btn-loading"></i>');
                n.prop("disabled", "disabled")
            } else {
                n.find(".ew-btn-loading").remove();
                n.removeProp("disabled", "disabled");
                if (n.find(".layui-icon.layui-hide").length <= 0) {
                    n.removeClass("icon-btn")
                }
                n.find(".layui-icon").removeClass("layui-hide");
                p && n.html(p)
            }
        },
        openSideAutoExpand: function () {
            h(".layui-layout-admin>.layui-side").off("mouseenter.openSideAutoExpand").on("mouseenter.openSideAutoExpand", function () {
                if (h(this).parent().hasClass("admin-nav-mini")) {
                    m.flexible(true);
                    h(this).addClass("side-mini-hover")
                }
            });
            h(".layui-layout-admin>.layui-side").off("mouseleave.openSideAutoExpand").on("mouseleave.openSideAutoExpand", function () {
                if (h(this).hasClass("side-mini-hover")) {
                    m.flexible(false);
                    h(this).removeClass("side-mini-hover")
                }
            })
        },
        openCellAutoExpand: function () {
            h("body").off("mouseenter.openCellAutoExpand").on("mouseenter.openCellAutoExpand", ".layui-table-view td", function () {
                h(this).find(".layui-table-grid-down").trigger("click")
            });
            h("body").off("mouseleave.openCellAutoExpand").on("mouseleave.openCellAutoExpand", ".layui-table-tips>.layui-layer-content", function () {
                h(".layui-table-tips-c").trigger("click")
            })
        },
        isTop: function () {
            return h(a).length > 0
        },
        strToWin: function (q) {
            var p = window;
            if (q) {
                var n = q.split(".");
                for (var o = 0; o < n.length; o++) {
                    p = p[n[o]]
                }
            }
            return p
        },
        parseLayerOption: function (p) {
            for (var q in p) {
                if (p[q] && p[q].toString().indexOf(",") != -1) {
                    p[q] = p[q].toString().split(",")
                }
            }
            var n = ["success", "cancel", "end", "full", "min", "restore"];
            for (var o = 0; o < n.length; o++) {
                for (var q in p) {
                    if (q == n[o]) {
                        p[q] = window[p[q]]
                    }
                }
            }
            if (p.content && (typeof p.content === "string") && p.content.indexOf("#") == 0) {
                p.content = h(p.content).html()
            }
            (p.type == undefined) && (p.type = 2);
            return p
        },
        ajaxSuccessBefore: function (n, o) {
            return true
        },
        getAjaxHeaders: function (n) {
            var o = new Array();
            return o
        }
    };
    m.events = {
        flexible: function () {
            m.strToWin(h(this).data("window")).layui.admin.flexible()
        },
        refresh: function () {
            m.strToWin(h(this).data("window")).layui.admin.refresh()
        },
        back: function () {
            m.strToWin(h(this).data("window")).history.back()
        },
        theme: function () {
            var n = h(this).data("url");
            m.strToWin(h(this).data("window")).layui.admin.popupRight({
                id: "layer-theme",
                type: 2,
                content: n ? n : "page/tpl/tpl-theme.html"
            })
        },
        note: function () {
            var n = h(this).data("url");
            m.strToWin(h(this).data("window")).layui.admin.popupRight({
                id: "layer-note",
                title: "便签",
                type: 2,
                closeBtn: false,
                content: n ? n : "page/tpl/tpl-note.html"
            })
        },
        message: function () {
            var n = h(this).data("url");
            m.strToWin(h(this).data("window")).layui.admin.popupRight({
                id: "layer-notice",
                type: 2,
                content: n ? n : "page/tpl/tpl-message.html"
            })
        },
        psw: function () {
            var n = h(this).data("url");
            m.strToWin(h(this).data("window")).layui.admin.open({
                id: "pswForm",
                type: 2,
                title: "修改密码",
                area: ["360px", "287px"],
                shade: 0,
                content: n ? n : "page/tpl/tpl-password.html"
            })
        },
        logout: function () {
            var n = h(this).data("url");
            m.strToWin(h(this).data("window")).layui.layer.confirm("确定要退出登录吗？", {
                title: "温馨提示",
                skin: "layui-layer-admin"
            }, function () {
                location.replace(n ? n : "/")
            })
        },
        open: function () {
            var n = h(this).data();
            m.strToWin(n.window).layui.admin.open(m.parseLayerOption(m.util.deepClone(n)))
        },
        popupRight: function () {
            var n = h(this).data();
            m.strToWin(n.window).layui.admin.popupRight(m.parseLayerOption(m.util.deepClone(n)))
        },
        fullScreen: function () {
            var u = "layui-icon-screen-full",
                n = "layui-icon-screen-restore";
            var r = h(this).find("i");
            var q = document.fullscreenElement || document.msFullscreenElement || document.mozFullScreenElement || document.webkitFullscreenElement || false;
            if (q) {
                var p = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
                if (p) {
                    p.call(document)
                } else {
                    if (window.ActiveXObject) {
                        var o = new ActiveXObject("WScript.Shell");
                        o && o.SendKeys("{F11}")
                    }
                }
                r.addClass(u).removeClass(n)
            } else {
                var s = document.documentElement;
                var t = s.requestFullscreen || s.webkitRequestFullscreen || s.mozRequestFullScreen || s.msRequestFullscreen;
                if (t) {
                    t.call(s)
                } else {
                    if (window.ActiveXObject) {
                        var o = new ActiveXObject("WScript.Shell");
                        o && o.SendKeys("{F11}")
                    }
                }
                r.addClass(n).removeClass(u)
            }
        },
        leftPage: function () {
            m.strToWin(h(this).data("window")).layui.admin.rollPage("left")
        },
        rightPage: function () {
            m.strToWin(h(this).data("window")).layui.admin.rollPage()
        },
        closeThisTabs: function () {
            var n = h(this).data("url");
            m.strToWin(h(this).data("window")).layui.admin.closeThisTabs(n)
        },
        closeOtherTabs: function () {
            m.strToWin(h(this).data("window")).layui.admin.closeOtherTabs()
        },
        closeAllTabs: function () {
            m.strToWin(h(this).data("window")).layui.admin.closeAllTabs()
        },
        closeDialog: function () {
            m.closeThisDialog()
        },
        closePageDialog: function () {
            m.closeDialog(this)
        }
    };
    m.chooseLocation = function (s) {
        var o = s.title;
        var w = s.onSelect;
        var q = s.needCity;
        var x = s.center;
        var A = s.defaultZoom;
        var t = s.pointZoom;
        var v = s.keywords;
        var z = s.pageSize;
        var r = s.mapJsUrl;
        (o == undefined) && (o = "选择位置");
        (A == undefined) && (A = 11);
        (t == undefined) && (t = 17);
        (v == undefined) && (v = "");
        (z == undefined) && (z = 30);
        (r == undefined) && (r = "https://webapi.amap.com/maps?v=1.4.14&key=006d995d433058322319fa797f2876f5");
        var B = false,
            y;
        var u = function (D, C) {
            AMap.service(["AMap.PlaceSearch"], function () {
                var F = new AMap.PlaceSearch({
                    type: "",
                    pageSize: z,
                    pageIndex: 1
                });
                var E = [C, D];
                F.searchNearBy(v, E, 1000, function (H, G) {
                    if (H == "complete") {
                        var K = G.poiList.pois;
                        var L = "";
                        for (var J = 0; J < K.length; J++) {
                            var I = K[J];
                            if (I.location != undefined) {
                                L += '<div data-lng="' + I.location.lng + '" data-lat="' + I.location.lat + '" class="ew-map-select-search-list-item">';
                                L += '     <div class="ew-map-select-search-list-item-title">' + I.name + "</div>";
                                L += '     <div class="ew-map-select-search-list-item-address">' + I.address + "</div>";
                                L += '     <div class="ew-map-select-search-list-item-icon-ok layui-hide"><i class="layui-icon layui-icon-ok-circle"></i></div>';
                                L += "</div>"
                            }
                        }
                        h("#ew-map-select-pois").html(L)
                    }
                })
            })
        };
        var n = function () {
            var C = {
                resizeEnable: true,
                zoom: A
            };
            x && (C.center = x);
            var D = new AMap.Map("ew-map-select-map", C);
            D.on("complete", function () {
                var E = D.getCenter();
                u(E.lat, E.lng)
            });
            D.on("moveend", function () {
                if (B) {
                    B = false
                } else {
                    h("#ew-map-select-tips").addClass("layui-hide");
                    h("#ew-map-select-center-img").removeClass("bounceInDown");
                    setTimeout(function () {
                        h("#ew-map-select-center-img").addClass("bounceInDown")
                    });
                    var E = D.getCenter();
                    u(E.lat, E.lng)
                }
            });
            h("#ew-map-select-pois").off("click").on("click", ".ew-map-select-search-list-item", function () {
                h("#ew-map-select-tips").addClass("layui-hide");
                h("#ew-map-select-pois .ew-map-select-search-list-item-icon-ok").addClass("layui-hide");
                h(this).find(".ew-map-select-search-list-item-icon-ok").removeClass("layui-hide");
                h("#ew-map-select-center-img").removeClass("bounceInDown");
                setTimeout(function () {
                    h("#ew-map-select-center-img").addClass("bounceInDown")
                });
                var G = h(this).data("lng");
                var H = h(this).data("lat");
                var F = h(this).find(".ew-map-select-search-list-item-title").text();
                var E = h(this).find(".ew-map-select-search-list-item-address").text();
                y = {
                    name: F,
                    address: E,
                    lat: H,
                    lng: G
                };
                B = true;
                D.setZoomAndCenter(t, [G, H])
            });
            h("#ew-map-select-btn-ok").click(function () {
                if (y == undefined) {
                    j.msg("请点击位置列表选择", {
                        icon: 2,
                        anim: 6
                    })
                } else {
                    if (w) {
                        if (q) {
                            var E = j.load(2);
                            D.setCenter([y.lng, y.lat]);
                            D.getCity(function (F) {
                                j.close(E);
                                y.city = F;
                                m.closeDialog("#ew-map-select-btn-ok");
                                w(y)
                            })
                        } else {
                            m.closeDialog("#ew-map-select-btn-ok");
                            w(y)
                        }
                    } else {
                        m.closeDialog("#ew-map-select-btn-ok")
                    }
                }
            });
            h("#ew-map-select-input-search").off("input").on("input", function () {
                var E = h(this).val();
                if (!E) {
                    h("#ew-map-select-tips").html("");
                    h("#ew-map-select-tips").addClass("layui-hide")
                }
                AMap.plugin("AMap.Autocomplete", function () {
                    var F = new AMap.Autocomplete({
                        city: "全国"
                    });
                    F.search(E, function (I, H) {
                        if (H.tips) {
                            var G = H.tips;
                            var K = "";
                            for (var J = 0; J < G.length; J++) {
                                var L = G[J];
                                if (L.location != undefined) {
                                    K += '<div data-lng="' + L.location.lng + '" data-lat="' + L.location.lat + '" class="ew-map-select-search-list-item">';
                                    K += '     <div class="ew-map-select-search-list-item-icon-search"><i class="layui-icon layui-icon-search"></i></div>';
                                    K += '     <div class="ew-map-select-search-list-item-title">' + L.name + "</div>";
                                    K += '     <div class="ew-map-select-search-list-item-address">' + L.address + "</div>";
                                    K += "</div>"
                                }
                            }
                            h("#ew-map-select-tips").html(K);
                            if (G.length == 0) {
                                h("#ew-map-select-tips").addClass("layui-hide")
                            } else {
                                h("#ew-map-select-tips").removeClass("layui-hide")
                            }
                        } else {
                            h("#ew-map-select-tips").html("");
                            h("#ew-map-select-tips").addClass("layui-hide")
                        }
                    })
                })
            });
            h("#ew-map-select-input-search").off("blur").on("blur", function () {
                var E = h(this).val();
                if (!E) {
                    h("#ew-map-select-tips").html("");
                    h("#ew-map-select-tips").addClass("layui-hide")
                }
            });
            h("#ew-map-select-input-search").off("focus").on("focus", function () {
                var E = h(this).val();
                if (E) {
                    h("#ew-map-select-tips").removeClass("layui-hide")
                }
            });
            h("#ew-map-select-tips").off("click").on("click", ".ew-map-select-search-list-item", function () {
                h("#ew-map-select-tips").addClass("layui-hide");
                var E = h(this).data("lng");
                var F = h(this).data("lat");
                y = undefined;
                D.setZoomAndCenter(t, [E, F])
            })
        };
        var p = '<div class="ew-map-select-tool" style="position: relative;">';
        p += '        搜索：<input id="ew-map-select-input-search" class="layui-input icon-search inline-block" style="width: 190px;" placeholder="输入关键字搜索" autocomplete="off" />';
        p += '        <button id="ew-map-select-btn-ok" class="layui-btn icon-btn pull-right" type="button"><i class="layui-icon">&#xe605;</i>确定</button>';
        p += '        <div id="ew-map-select-tips" class="ew-map-select-search-list layui-hide">';
        p += "        </div>";
        p += "   </div>";
        p += '   <div class="layui-row ew-map-select">';
        p += '        <div class="layui-col-sm7 ew-map-select-map-group" style="position: relative;">';
        p += '             <div id="ew-map-select-map"></div>';
        p += '             <i id="ew-map-select-center-img2" class="layui-icon layui-icon-add-1"></i>';
        p += '             <img id="ew-map-select-center-img" src="https://3gimg.qq.com/lightmap/components/locationPicker2/image/marker.png"/>';
        p += "        </div>";
        p += '        <div id="ew-map-select-pois" class="layui-col-sm5 ew-map-select-search-list">';
        p += "        </div>";
        p += "   </div>";
        m.open({
            id: "ew-map-select",
            type: 1,
            title: o,
            area: "750px",
            content: p,
            success: function (C, E) {
                var D = h(C).children(".layui-layer-content");
                D.css("overflow", "visible");
                m.showLoading(D);
                if (undefined == window.AMap) {
                    h.getScript(r, function () {
                        n();
                        m.removeLoading(D)
                    })
                } else {
                    n();
                    m.removeLoading(D)
                }
            }
        })
    };
    m.cropImg = function (q) {
        var o = "image/jpeg";
        var v = q.aspectRatio;
        var w = q.imgSrc;
        var t = q.imgType;
        var r = q.onCrop;
        var s = q.limitSize;
        var u = q.acceptMime;
        var p = q.exts;
        var n = q.title;
        (v == undefined) && (v = 1 / 1);
        (n == undefined) && (n = "裁剪图片");
        t && (o = t);
        layui.use(["Cropper", "upload"], function () {
            var y = layui.Cropper;
            var x = layui.upload;

            function z() {
                var C, D = h("#ew-crop-img");
                var E = {
                    elem: "#ew-crop-img-upload",
                    auto: false,
                    drag: false,
                    choose: function (F) {
                        F.preview(function (H, I, G) {
                            o = I.type;
                            D.attr("src", G);
                            if (!w || !C) {
                                w = G;
                                z()
                            } else {
                                C.destroy();
                                C = new y(D[0], B)
                            }
                        })
                    }
                };
                (s != undefined) && (E.size = s);
                (u != undefined) && (E.acceptMime = u);
                (p != undefined) && (E.exts = p);
                x.render(E);
                if (!w) {
                    h("#ew-crop-img-upload").trigger("click");
                    return
                }
                var B = {
                    aspectRatio: v,
                    preview: "#ew-crop-img-preview"
                };
                C = new y(D[0], B);
                h(".ew-crop-tool").on("click", "[data-method]", function () {
                    var G = h(this).data(),
                        H, F;
                    if (!C || !G.method) {
                        return
                    }
                    G = h.extend({}, G);
                    H = C.cropped;
                    switch (G.method) {
                        case "rotate":
                            if (H && B.viewMode > 0) {
                                C.clear()
                            }
                            break;
                        case "getCroppedCanvas":
                            if (o === "image/jpeg") {
                                if (!G.option) {
                                    G.option = {}
                                }
                                G.option.fillColor = "#fff"
                            }
                            break
                    }
                    F = C[G.method](G.option, G.secondOption);
                    switch (G.method) {
                        case "rotate":
                            if (H && B.viewMode > 0) {
                                C.crop()
                            }
                            break;
                        case "scaleX":
                        case "scaleY":
                            h(this).data("option", -G.option);
                            break;
                        case "getCroppedCanvas":
                            if (F) {
                                r && r(F.toDataURL(o));
                                m.closeDialog("#ew-crop-img")
                            } else {
                                j.msg("裁剪失败", {
                                    icon: 2,
                                    anim: 6
                                })
                            }
                            break
                    }
                })
            }

            var A = '<div class="layui-row">';
            A += '        <div class="layui-col-sm8" style="min-height: 9rem;">';
            A += '             <img id="ew-crop-img" src="' + (w ? w : "") + '" style="max-width:100%;" />';
            A += "        </div>";
            A += '        <div class="layui-col-sm4 layui-hide-xs" style="padding: 0 20px;text-align: center;">';
            A += '             <div id="ew-crop-img-preview" style="width: 100%;height: 9rem;overflow: hidden;display: inline-block;border: 1px solid #dddddd;"></div>';
            A += "        </div>";
            A += "   </div>";
            A += '   <div class="text-center ew-crop-tool" style="padding: 15px 10px 5px 0;">';
            A += '        <div class="layui-btn-group" style="margin-bottom: 10px;margin-left: 10px;">';
            A += '             <button title="放大" data-method="zoom" data-option="0.1" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-add-1"></i></button>';
            A += '             <button title="缩小" data-method="zoom" data-option="-0.1" class="layui-btn icon-btn" type="button"><span style="display: inline-block;width: 12px;height: 2.5px;background: rgba(255, 255, 255, 0.9);vertical-align: middle;margin: 0 4px;"></span></button>';
            A += "        </div>";
            A += '        <div class="layui-btn-group layui-hide-xs" style="margin-bottom: 10px;">';
            A += '             <button title="向左旋转" data-method="rotate" data-option="-45" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-refresh-1" style="transform: rotateY(180deg) rotate(40deg);display: inline-block;"></i></button>';
            A += '             <button title="向右旋转" data-method="rotate" data-option="45" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-refresh-1" style="transform: rotate(30deg);display: inline-block;"></i></button>';
            A += "        </div>";
            A += '        <div class="layui-btn-group" style="margin-bottom: 10px;">';
            A += '             <button title="左移" data-method="move" data-option="-10" data-second-option="0" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-left"></i></button>';
            A += '             <button title="右移" data-method="move" data-option="10" data-second-option="0" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-right"></i></button>';
            A += '             <button title="上移" data-method="move" data-option="0" data-second-option="-10" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-up"></i></button>';
            A += '             <button title="下移" data-method="move" data-option="0" data-second-option="10" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-down"></i></button>';
            A += "        </div>";
            A += '        <div class="layui-btn-group" style="margin-bottom: 10px;">';
            A += '             <button title="左右翻转" data-method="scaleX" data-option="-1" class="layui-btn icon-btn" type="button" style="position: relative;width: 41px;"><i class="layui-icon layui-icon-triangle-r" style="position: absolute;left: 9px;top: 0;transform: rotateY(180deg);font-size: 16px;"></i><i class="layui-icon layui-icon-triangle-r" style="position: absolute; right: 3px; top: 0;font-size: 16px;"></i></button>';
            A += '             <button title="上下翻转" data-method="scaleY" data-option="-1" class="layui-btn icon-btn" type="button" style="position: relative;width: 41px;"><i class="layui-icon layui-icon-triangle-d" style="position: absolute;left: 11px;top: 6px;transform: rotateX(180deg);line-height: normal;font-size: 16px;"></i><i class="layui-icon layui-icon-triangle-d" style="position: absolute; left: 11px; top: 14px;line-height: normal;font-size: 16px;"></i></button>';
            A += "        </div>";
            A += '        <div class="layui-btn-group" style="margin-bottom: 10px;">';
            A += '             <button title="重新开始" data-method="reset" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-refresh"></i></button>';
            A += '             <button title="选择图片" id="ew-crop-img-upload" class="layui-btn icon-btn" type="button"><i class="layui-icon layui-icon-upload-drag"></i></button>';
            A += "        </div>";
            A += '        <button data-method="getCroppedCanvas" data-option="{ &quot;maxWidth&quot;: 4096, &quot;maxHeight&quot;: 4096 }" class="layui-btn icon-btn" type="button" style="margin-left: 10px;margin-bottom: 10px;"><i class="layui-icon">&#xe605;</i>完成</button>';
            A += "   </div>";
            m.open({
                title: n,
                area: "665px",
                type: 1,
                content: A,
                success: function (B, C) {
                    h(B).children(".layui-layer-content").css("overflow", "visible");
                    z()
                }
            })
        })
    };
    m.util = {
        Convert_BD09_To_GCJ02: function (o) {
            var q = (3.141592653589793 * 3000) / 180;
            var n = o.lng - 0.0065,
                s = o.lat - 0.006;
            var r = Math.sqrt(n * n + s * s) - 0.00002 * Math.sin(s * q);
            var p = Math.atan2(s, n) - 0.000003 * Math.cos(n * q);
            o.lng = r * Math.cos(p);
            o.lat = r * Math.sin(p);
            return o
        },
        Convert_GCJ02_To_BD09: function (o) {
            var q = (3.141592653589793 * 3000) / 180;
            var n = o.lng,
                s = o.lat;
            var r = Math.sqrt(n * n + s * s) + 0.00002 * Math.sin(s * q);
            var p = Math.atan2(s, n) + 0.000003 * Math.cos(n * q);
            o.lng = r * Math.cos(p) + 0.0065;
            o.lat = r * Math.sin(p) + 0.006;
            return o
        },
        animateNum: function (D, x, F, v) {
            var r = h(D);
            var s = r.text().replace(/,/g, "");
            x = x === null || x === undefined || x === true || x === "true";
            F = isNaN(F) ? 500 : F;
            v = isNaN(v) ? 100 : v;
            var z = "INPUT,TEXTAREA".indexOf(r.get(0).tagName) >= 0;
            var t = function (J) {
                    var H = "";
                    for (var I = 0; I < J.length; I++) {
                        if (!isNaN(J.charAt(I))) {
                            return H
                        } else {
                            H += J.charAt(I)
                        }
                    }
                },
                A = function (J) {
                    var H = "";
                    for (var I = J.length - 1; I >= 0; I--) {
                        if (!isNaN(J.charAt(I))) {
                            return H
                        } else {
                            H = J.charAt(I) + H
                        }
                    }
                },
                C = function (I, H) {
                    if (!H) {
                        return I
                    }
                    if (!/^[0-9]+.?[0-9]*$/.test(I)) {
                        return I
                    }
                    I = I.toString();
                    return I.replace(I.indexOf(".") > 0 ? /(\d)(?=(\d{3})+(?:\.))/g : /(\d)(?=(\d{3})+(?:$))/g, "$1,")
                };
            var G = t(s.toString());
            var p = A(s.toString());
            var q = s.toString().replace(G, "").replace(p, "");
            if (isNaN(q) || q === 0) {
                z ? r.val(s) : r.html(s);
                console.error("非法数值！");
                return
            }
            var u = q.split(".");
            var o = u[1] ? u[1].length : 0;
            var n = 0,
                y = q;
            if (Math.abs(y) > 10) {
                n = parseFloat(u[0].substring(0, u[0].length - 1) + (u[1] ? ".0" + u[1] : ""))
            }
            var w = (y - n) / v,
                E = 0;
            var B = setInterval(function () {
                var H = G + C(n.toFixed(o), x) + p;
                z ? r.val(H) : r.html(H);
                n += w;
                E++;
                if (Math.abs(n) >= Math.abs(y) || E > 5000) {
                    H = G + C(y, x) + p;
                    z ? r.val(H) : r.html(H);
                    clearInterval(B)
                }
            }, F / v)
        },
        deepClone: function (q) {
            var n;
            var o = m.util.isClass(q);
            if (o === "Object") {
                n = {}
            } else {
                if (o === "Array") {
                    n = []
                } else {
                    return q
                }
            }
            for (var p in q) {
                var r = q[p];
                if (m.util.isClass(r) == "Object") {
                    n[p] = arguments.callee(r)
                } else {
                    if (m.util.isClass(r) == "Array") {
                        n[p] = arguments.callee(r)
                    } else {
                        n[p] = q[p]
                    }
                }
            }
            return n
        },
        isClass: function (n) {
            if (n === null) {
                return "Null"
            }
            if (n === undefined) {
                return "Undefined"
            }
            return Object.prototype.toString.call(n).slice(8, -1)
        },
        fullTextIsEmpty: function (q) {
            if (!q) {
                return true
            }
            var o = ["img", "audio", "video", "iframe", "object"];
            for (var n = 0; n < o.length; n++) {
                if (q.indexOf("<" + o[n]) > -1) {
                    return false
                }
            }
            var p = q.replace(/\s*/g, "");
            if (!p) {
                return true
            }
            p = p.replace(/&nbsp;/ig, "");
            if (!p) {
                return true
            }
            p = p.replace(/<[^>]+>/g, "");
            if (!p) {
                return true
            }
            return false
        }
    };
    var l = ".layui-layout-admin.admin-nav-mini>.layui-side .layui-nav .layui-nav-item";
    h(document).on("mouseenter", l + "," + l + " .layui-nav-child>dd", function () {
        if (m.getPageWidth() > 768) {
            var o = h(this),
                q = o.find(">.layui-nav-child");
            if (q.length > 0) {
                o.addClass("admin-nav-hover");
                q.css("left", o.offset().left + o.outerWidth());
                var p = o.offset().top;
                if (p + q.outerHeight() > m.getPageHeight()) {
                    p = p - q.outerHeight() + o.outerHeight();
                    (p < 60) && (p = 60);
                    q.addClass("show-top")
                }
                q.css("top", p);
                q.addClass("ew-anim-drop-in")
            } else {
                if (o.hasClass("layui-nav-item")) {
                    var n = o.find("cite").text();
                    j.tips(n, o, {
                        tips: [2, "#303133"],
                        time: -1,
                        success: function (r, s) {
                            h(r).css("margin-top", "12px")
                        }
                    })
                }
            }
        }
    }).on("mouseleave", l + "," + l + " .layui-nav-child>dd", function () {
        j.closeAll("tips");
        var o = h(this);
        o.removeClass("admin-nav-hover");
        var n = o.find(">.layui-nav-child");
        n.removeClass("show-top ew-anim-drop-in");
        n.css({
            "left": "unset",
            "top": "unset"
        })
    });
    h(document).on("click", "*[ew-event]", function () {
        var n = h(this).attr("ew-event");
        var o = m.events[n];
        o && o.call(this, h(this))
    });
    h(document).on("mouseenter", "*[lay-tips]", function () {
        var n = h(this).attr("lay-tips");
        var o = h(this).attr("lay-direction");
        var p = h(this).attr("lay-bg");
        var q = h(this).attr("lay-offset");
        j.tips(n, this, {
            tips: [o || 1, p || "#303133"],
            time: -1,
            success: function (r, s) {
                if (q) {
                    q = q.split(",");
                    var u = q[0],
                        t = q.length > 1 ? q[1] : undefined;
                    u && (h(r).css("margin-top", u));
                    t && (h(r).css("margin-left", t))
                }
            }
        })
    }).on("mouseleave", "*[lay-tips]", function () {
        j.closeAll("tips")
    });
    if (m.getPageWidth() < 768) {
        if (layui.device().os == "windows") {
            h("body").append("<style>@media screen and (max-width: 768px) {::-webkit-scrollbar{width:8px;height:9px;background:transparent}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{border-radius:5px;background-color:#c1c1c1}::-webkit-scrollbar-thumb:hover{background-color:#a8a8a8}.mini-bar::-webkit-scrollbar{width:5px;height:5px}.mini-bar::-webkit-scrollbar-thumb{border-radius:3px}}</style>")
        }
    }
    h(document).on("click", "*[ew-href]", function () {
        var n = h(this).attr("ew-href");
        var o = h(this).attr("ew-title");
        o || (o = h(this).text());
        if (top.layui && top.layui.index) {
            top.layui.index.openTab({
                title: o ? o : "",
                url: n
            })
        } else {
            location.href = n
        }
    });
    var g = layui.data(m.tableName);
    if (g && g.theme) {
        (g.theme == c) || layui.link(m.getThemeDir() + g.theme + m.getCssSuffix(), g.theme)
    } else {
        if (c != m.defaultTheme) {
            layui.link(m.getThemeDir() + m.defaultTheme + m.getCssSuffix(), m.defaultTheme)
        }
    }
    if (!layui.contextMenu) {
        h(document).off("click.ctxMenu").on("click.ctxMenu", function () {
            try {
                var q = top.window.frames;
                for (var n = 0; n < q.length; n++) {
                    var o = q[n];
                    try {
                        (o.layui && o.layui.jquery) && o.layui.jquery("body>.ctxMenu").remove()
                    } catch (p) {
                    }
                }
                try {
                    (top.layui && top.layui.jquery) && top.layui.jquery("body>.ctxMenu").remove()
                } catch (p) {
                }
            } catch (p) {
            }
        })
    }
    f("admin", m)
});