package models;

import java.util.HashMap;

public enum Tools {
    CHEMSPOT("chemspot"),
    CHENER("chener"),
    CHEMICAL_DATA_EXTRACTOR("chemde"),
    CHATGPT("chatgpt");

    private final String value;
    private static final HashMap<String, Tools> NameToTool = new HashMap<>();

    static {
        for (Tools e : values()) {
            NameToTool.put(e.value, e);
        }
    }

    Tools(String name) {
        this.value = name;
    }

    public static Tools ValueOf(String str){
        return NameToTool.get(str);
    }
}
