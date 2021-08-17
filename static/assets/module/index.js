﻿/** EasyWeb iframe v3.1.5 date:2019-10-05 License By http://easyweb.vip */
layui.define(["layer", "element", "admin"], function (s) {
    var d = layui.jquery;
    var r = layui.layer;
    var b = layui.element;
    var m = layui.admin;
    var a = ".layui-layout-admin>.layui-header";
    var o = ".layui-layout-admin>.layui-side>.layui-side-scroll";
    var j = ".layui-layout-admin>.layui-body";
    var n = j + ">.layui-tab";
    var q = j + ">.layui-body-header";
    var i = "admin-pagetabs";
    var p = "admin-side-nav";
    var k = {};
    var f = false;
    var c;
    var h = {
        pageTabs: true,
        cacheTab: true,
        openTabCtxMenu: true,
        maxTabNum: 20,
        mTabList: [],
        mTabPosition: undefined,
        loadView: function (y) {
            var w = y.menuPath;
            var v = y.menuName;
            if (!w) {
                console.error("url不能为空");
                r.msg("url不能为空", {
                    icon: 2
                });
                return
            }
            if (h.pageTabs) {
                var u = false;
                d(n + ">.layui-tab-title>li").each(function () {
                    if (d(this).attr("lay-id") === w) {
                        u = true;
                        return false
                    }
                });
                if (!u) {
                    if ((h.mTabList.length + 1) >= h.maxTabNum) {
                        r.msg("最多打开" + h.maxTabNum + "个选项卡", {
                            icon: 2
                        });
                        m.activeNav(h.mTabPosition);
                        return
                    }
                    f = true;
                    b.tabAdd(i, {
                        id: w,
                        title: '<span class="title">' + (v ? v : "") + "</span>",
                        content: '<iframe lay-id="' + w + '" src="' + w + '" frameborder="0" class="admin-iframe"></iframe>'
                    });
                    if (w != c) {
                        h.mTabList.push(y)
                    }
                    if (h.cacheTab) {
                        m.putTempData("indexTabs", h.mTabList)
                    }
                }
                b.tabChange(i, w)
            } else {
                var t = d(j + ">.admin-iframe");
                if (!t || t.length <= 0) {
                    var x = '<div class="layui-body-header">';
                    x += '      <span class="layui-body-header-title"></span>';
                    x += '      <span class="layui-breadcrumb pull-right">';
                    x += '         <a ew-href="' + c + '">首页</a>';
                    x += "         <a><cite></cite></a>";
                    x += "      </span>";
                    x += "   </div>";
                    x += '   <div style="-webkit-overflow-scrolling: touch;">';
                    x += '      <iframe lay-id="' + w + '" src="' + w + '" frameborder="0" class="admin-iframe"></iframe>';
                    x += "   </div>";
                    d(j).html(x);
                    if (w != c) {
                        h.setTabTitle(v)
                    }
                    b.render("breadcrumb")
                } else {
                    t.attr("lay-id", w);
                    t.attr("src", w);
                    h.setTabTitle(v)
                }
                m.activeNav(w);
                h.mTabList.splice(0, h.mTabList.length);
                if (w != c) {
                    h.mTabList.push(y);
                    h.mTabPosition = w
                } else {
                    h.mTabPosition = undefined
                }
                if (h.cacheTab) {
                    m.putTempData("indexTabs", h.mTabList);
                    m.putTempData("tabPosition", h.mTabPosition)
                }
            }
            if (m.getPageWidth() <= 768) {
                m.flexible(true)
            }
        },
        loadHome: function (v) {
            c = v.menuPath;
            var w = m.getTempData("indexTabs");
            var t = m.getTempData("tabPosition");
            var u = (v.loadSetting == undefined ? true : v.loadSetting);
            h.loadView({
                menuPath: c,
                menuName: v.menuName
            });
            if (!h.pageTabs) {
                m.activeNav(v.menuPath)
            }
            if (u) {
                h.loadSettings(w, t, v.onlyLast)
            }
        },
        openTab: function (v) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.index) {
                    top.layui.index.openTab(v);
                    return
                }
            }
            var t = v.url;
            var u = v.title;
            if (v.end) {
                k[t] = v.end
            }
            h.loadView({
                menuPath: t,
                menuName: u
            })
        },
        closeTab: function (t) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.index) {
                    top.layui.index.closeTab(t);
                    return
                }
            }
            b.tabDelete(i, t)
        },
        loadSettings: function (z, y, w) {
            if (h.cacheTab) {
                var A = z;
                var v = y;
                if (A) {
                    var u = -1;
                    for (var x = 0; x < A.length; x++) {
                        if (h.pageTabs && !w) {
                            h.loadView(A[x])
                        }
                        if (A[x].menuPath == v) {
                            u = x
                        }
                    }
                    if (u != -1) {
                        setTimeout(function () {
                            h.loadView(A[u]);
                            if (!h.pageTabs) {
                                m.activeNav(v)
                            }
                        }, 150)
                    }
                }
            }
            var t = layui.data(m.tableName);
            if (t) {
                if (t.openFooter != undefined && t.openFooter == false) {
                    d("body.layui-layout-body").addClass("close-footer")
                }
                if (t.tabAutoRefresh) {
                    d(n).attr("lay-autoRefresh", "true")
                }
                if (t.navArrow != undefined) {
                    d(o + ">.layui-nav-tree").removeClass("arrow2 arrow3");
                    t.navArrow && d(o + ">.layui-nav-tree").addClass(t.navArrow)
                }
            }
        },
        setTabCache: function (t) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.index) {
                    top.layui.index.setTabCache(t);
                    return
                }
            }
            layui.data(m.tableName, {
                key: "cacheTab",
                value: t
            });
            h.cacheTab = t;
            if (t) {
                m.putTempData("indexTabs", h.mTabList);
                m.putTempData("tabPosition", h.mTabPosition)
            } else {
                m.putTempData("indexTabs", []);
                m.putTempData("tabPosition", undefined)
            }
        },
        clearTabCache: function () {
            m.putTempData("indexTabs", undefined)
        },
        setTabTitle: function (u, t) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.index) {
                    top.layui.index.setTabTitle(u, t);
                    return
                }
            }
            if (!h.pageTabs) {
                if (u) {
                    d(q).addClass("show");
                    var v = d(q + ">.layui-body-header-title");
                    v.html(u);
                    v.next(".layui-breadcrumb").find("cite").last().text(u)
                } else {
                    d(q).removeClass("show")
                }
            } else {
                u || (u = "");
                t || (t = d(n + ">.layui-tab-title>li.layui-this").attr("lay-id"));
                t && d(n + '>.layui-tab-title>li[lay-id="' + t + '"] .title').html(u)
            }
        },
        setTabTitleHtml: function (t) {
            if (window != top && !m.isTop()) {
                if (top.layui && top.layui.index) {
                    top.layui.index.setTabTitleHtml(t);
                    return
                }
            }
            if (!h.pageTabs) {
                if (t) {
                    d(q).addClass("show");
                    d(q).html(t)
                } else {
                    d(q).removeClass("show")
                }
            }
        },
        closeTabCache: function () {
            console.warn("closeTabCache() has been deprecated, please use clearTabCache().");
            h.clearTabCache()
        },
        loadSetting: function () {
            console.warn("loadSetting() has been deprecated.")
        }
    };
    var l = layui.data(m.tableName);
    if (l) {
        if (l.openTab != undefined) {
            h.pageTabs = l.openTab
        }
        if (l.cacheTab != undefined) {
            h.cacheTab = l.cacheTab
        }
    }
    var g = ".layui-layout-admin .site-mobile-shade";
    if (d(g).length <= 0) {
        d(".layui-layout-admin").append('<div class="site-mobile-shade"></div>')
    }
    d(g).click(function () {
        m.flexible(true)
    });
    if (h.pageTabs && d(n).length <= 0) {
        var e = '<div class="layui-tab" lay-allowClose="true" lay-filter="admin-pagetabs">';
        e += '       <ul class="layui-tab-title"></ul>';
        e += '      <div class="layui-tab-content"></div>';
        e += "   </div>";
        e += '   <div class="layui-icon admin-tabs-control layui-icon-prev" ew-event="leftPage"></div>';
        e += '   <div class="layui-icon admin-tabs-control layui-icon-next" ew-event="rightPage"></div>';
        e += '   <div class="layui-icon admin-tabs-control layui-icon-down">';
        e += '      <ul class="layui-nav admin-tabs-select" lay-filter="admin-pagetabs-nav">';
        e += '         <li class="layui-nav-item" lay-unselect>';
        e += "            <a></a>";
        e += '            <dl class="layui-nav-child layui-anim-fadein">';
        e += '               <dd ew-event="closeThisTabs" lay-unselect><a>关闭当前标签页</a></dd>';
        e += '               <dd ew-event="closeOtherTabs" lay-unselect><a>关闭其它标签页</a></dd>';
        e += '               <dd ew-event="closeAllTabs" lay-unselect><a>关闭全部标签页</a></dd>';
        e += "            </dl>";
        e += "         </li>";
        e += "      </ul>";
        e += "   </div>";
        d(j).html(e);
        b.render("nav")
    }
    b.on("nav(" + p + ")", function (w) {
        var v = d(w);
        var t = v.attr("lay-href");
        var x = v.attr("lay-id");
        if (!x) {
            x = t
        }
        if (t && t != "javascript:;") {
            var u = v.attr("ew-title");
            u || (u = v.text().replace(/(^\s*)|(\s*$)/g, ""));
            h.loadView({
                menuId: x,
                menuPath: t,
                menuName: u
            })
        }
    });
    b.on("tab(" + i + ")", function (v) {
        var u = d(this).attr("lay-id");
        if (u != c) {
            h.mTabPosition = u
        } else {
            h.mTabPosition = undefined
        }
        if (h.cacheTab) {
            m.putTempData("tabPosition", h.mTabPosition)
        }
        m.rollPage("auto");
        m.activeNav(u);
        var t = d(n).attr("lay-autoRefresh");
        if (t === "true" && !f) {
            m.refresh(u)
        }
        f = false
    });
    b.on("tabDelete(" + i + ")", function (v) {
        var t = h.mTabList[v.index - 1];
        if (t) {
            var u = t.menuPath;
            h.mTabList.splice(v.index - 1, 1);
            if (h.cacheTab) {
                m.putTempData("indexTabs", h.mTabList)
            }
            if (k[u]) {
                k[u].call()
            }
        }
        if (d(n + ">.layui-tab-title>li.layui-this").length <= 0) {
            d(n + ">.layui-tab-title>li:last").trigger("click")
        }
    });
    d(document).off("click.navMore").on("click.navMore", "[nav-bind]", function () {
        var t = d(this).attr("nav-bind");
        d('ul[lay-filter="' + p + '"]').addClass("layui-hide");
        d('ul[nav-id="' + t + '"]').removeClass("layui-hide");
        if (m.getPageWidth() <= 768) {
            m.flexible(false)
        }
        d(a + ">.layui-nav .layui-nav-item").removeClass("layui-this");
        d(this).parent(".layui-nav-item").addClass("layui-this")
    });
    if (h.openTabCtxMenu && h.pageTabs) {
        layui.use("contextMenu", function () {
            var t = layui.contextMenu;
            if (t) {
                d(n + ">.layui-tab-title").off("contextmenu.tab").on("contextmenu.tab", "li", function (v) {
                    var u = d(this).attr("lay-id");
                    t.show([{
                        icon: "layui-icon layui-icon-refresh",
                        name: "刷新当前",
                        click: function () {
                            b.tabChange(i, u);
                            var w = d(n).attr("lay-autoRefresh");
                            if (!w || w !== "true") {
                                m.refresh(u)
                            }
                        }
                    }, {
                        icon: "layui-icon layui-icon-close-fill ctx-ic-lg",
                        name: "关闭当前",
                        click: function () {
                            m.closeThisTabs(u)
                        }
                    }, {
                        icon: "layui-icon layui-icon-unlink",
                        name: "关闭其他",
                        click: function () {
                            m.closeOtherTabs(u)
                        }
                    }, {
                        icon: "layui-icon layui-icon-close ctx-ic-lg",
                        name: "关闭全部",
                        click: function () {
                            m.closeAllTabs()
                        }
                    }], v.clientX, v.clientY);
                    return false
                })
            }
        })
    }
    s("index", h)
});