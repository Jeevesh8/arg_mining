import bs4
import os
import re
from bs4 import BeautifulSoup
from typing import Optional

footnote_regex = r"(\n\&gt;\*Hello[\S]*)"
url_regex = r"(https?://)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*(\.html|\.htm)*"
quote_regex = r"\&gt;(.*)\n"


def clean_text(text):
    """Replaces HTML specific punctuations and symbols to their commonly occuring counterparts.
    Adds spaces around punctuations(excluding >, < symbols, which are used around xml/html tags).
    """
    #    print("Before replaces:", text)
    replaces = [("’", "'"), ("“", '"'), ("”", '"'), ("&", "and"),
                ("∆", "[DELTA]")]
    for elem in replaces:
        text = text.replace(*elem)

    for elem in [".", ",", "!", ";", ":", "*", "?", "/", '"', "(", ")", "^"]:
        text = text.replace(elem, " " + elem + " ")

    #    print("After replaces:", text)
    return text


def add_tags(post, user_dict):
    """Adds user, url, and quote tags. Adds spaces around <claim>, </claim>, <premise>, </premise> tags.
    Additionally tries to remove away some footnotes.
    Args:
        post:       The text of a post, having claim, premise tags as occuring in the original xml file.
        user_dict:  A dict of already existing users for the current thread. Maintain one dict per thread.
    Returns:
        The post with the modifications and the user tag(str of for [USERi]).
    """
    if post["author"] not in user_dict:
        user_dict[post["author"]] = len(user_dict)

    text = str(post)  # Note 2

    user_tag = "[USER" + str(user_dict[post["author"]]) + "]"

    for elem in [
        ("</claim>", "</claim> "),
        ("<claim", " <claim"),
        ("<premise", " <premise"),
        ("</premise>", "</premise> "),
    ]:
        text = text.replace(*elem)

    text = re.sub(footnote_regex, "", text)
    text = re.sub(url_regex, "[URL]", text)
    text = re.sub(quote_regex, "[STARTQ]" + r"\1" + "[ENDQ] ", text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    return text, user_tag


def get_components(
    component: bs4.BeautifulSoup,
    parent_type: Optional[str] = None,
    parent_id: Optional[str] = None,
    parent_refers: Optional[str] = None,
    parent_rel_type: Optional[str] = None,
):
    """
    Args:
        component:       A parsed component, with 0 to any number of nested <claim> / <premise> tags

        parent_type:     The type of immediatel enclosing tag of the component. If the initial component is "<claim1> abc <premise> bfd </premise> ghj </claim1>"
                         its immediate enclosing type is None, as the recursion rolls over all the components in <claim> tags, viz. "abc", "<premise> bfd </premise>",
                         "ghj"; all of their parent_type are claim, when recursion rolls inside <premise> tags, over "bfd" its parent type is premise.The three of
                         these are yielded as distinct components. Let "abc", "ghj" be called the "stubs" of the current component.(The parts directly under
                         the outermost tags.)

        parent_id:       The id of the immediate enclosing tags of a component. Initially None. The id of "abc" is same as id of claim1 tags,
                         the id of "ghj" is id of claim1 tag + "Ć", similarly more "Ć" appended for following stubs of the current component.

        parent_refers:   The component referred by the immediate enclosing tag of the current component. Initially None. The parent_refers of "abc" is the "ref"
                         attribute of claim1 tags, the parent of "ghj" is the id of previous("abc") stub, similarly each stub refers to its previous one.

        parent_rel_type: The relation type with which the immediate enclosing tag is connected to parent_refers. Initially None. Propagated similar to
                         parent_refers, with relation type "continue". First stub's relation type same as immediate enclosing tag's.

    Yields:
        Mested components from a parsed component one-by-one. In the form (text, type, id, refers, relation_type) where:
            text:          The text of the component
            type:          other/claim/premise
            id:            The id of the component, None for Non-Argumentative
            refers:        The ids of the component the current component is related to. None, if it isn't related to any component(separated with _)
            relation_type: The type of relation between current component and refers. None, iff refers is None.
    """
    def is_stub(part):
        return not str(part).strip().startswith("<claim") and not str(
            part).strip().startswith("<premise")

    def is_first_stub(parts, part):
        for elem in parts:
            if is_stub(str(elem).strip()):
                return elem == part

    def chain_yield(comp_type="claim"):
        """Recursion; only entered when component has an enclosing tag like <claim> or <premise>.
        Otherwise, the component is returned with parent_id, parent_refers and parent_rel_type.
        For each part in the current component, calls get_components() with the appropriate parent_*
        attributes.
        """
        nonlocal component
        component = str(component)
        parsed_component = BeautifulSoup(component, "xml")

        # Get Top Most Tag's Attributes
        parent_id = str(parsed_component.find(comp_type)["id"])
        try:
            parent_refers = str(parsed_component.find(comp_type)["ref"])
            parent_rel_type = str(parsed_component.find(comp_type)["rel"])
        except KeyError:
            parent_refers = None
            parent_rel_type = None

        stub_parent_id, stub_parent_refers, stub_parent_rel_type = (
            parent_id,
            parent_refers,
            parent_rel_type,
        )
        for part in parsed_component.find(comp_type).contents:

            if is_stub(part):
                if not is_first_stub(
                        parsed_component.find(comp_type).contents, part):
                    stub_parent_refers = stub_parent_id
                    stub_parent_id += "Ć"
                    stub_parent_rel_type = "cont"

                for _ in get_components(
                        part,
                        comp_type,
                        stub_parent_id,
                        stub_parent_refers,
                        stub_parent_rel_type,
                ):
                    yield _
            else:
                for _ in get_components(part, comp_type, parent_id,
                                        parent_refers, parent_rel_type):
                    yield _

    if str(component).strip() == "":
        yield None

    elif str(component).strip().startswith("<claim"):
        for _ in chain_yield(comp_type="claim"):
            yield _

    elif str(component).strip().startswith("<premise"):
        for _ in chain_yield(comp_type="premise"):
            yield _

    else:
        if clean_text(str(component).strip()) == "":
            print("Component reduced to empty by cleaning: ", str(component))
        yield (
            clean_text(str(component).strip()),
            "other" if parent_type is None else parent_type,
            parent_id,
            parent_refers,
            parent_rel_type,
        )


def generate_components(filename):
    """Yields components from a thread one-by-one. In the form:
    (text, type, id, refers, relation_type)
    text:          The text of the component
    type:          other/claim/premise
    id:            The id of the component, None for Non-Argumentative
    refers:        The ids of the component the current component is related to. None, if it isn't related to any component(separated with _)
    relation_type: The type of relation between current component and refers. None, iff refers is None.
    """

    with open(filename, "r") as g:
        xml_str = g.read().replace("&#8217;", "&")

    xml_with_html_substituted = str(BeautifulSoup(xml_str, "lxml"))
    parsed_xml = BeautifulSoup(xml_with_html_substituted, "xml")  # Note 1.

    if len(re.findall(r"\&\#.*;", str(parsed_xml))) != 0:
        raise AssertionError("HTML characters still remaining in XML: " +
                             str(re.findall(r"\&\#.*;", str(parsed_xml))))

    user_dict = {}

    yield (parsed_xml.find("title").get_text(), "claim", "title", None, None)

    for post in [parsed_xml.find("op")] + parsed_xml.find_all("reply"):
        modified_post, user_tag = add_tags(post, user_dict)
        parsed_modified_post = BeautifulSoup(modified_post, "xml")

        try:
            contents = parsed_modified_post.find("reply").contents
        except AttributeError:
            contents = parsed_modified_post.find("op").contents

        yield (user_tag, "user_tag", None, None, None)

        for component in contents:
            for elem in get_components(component):
                if elem is not None:
                    yield elem


def get_all_threads():
    for t in ["negative", "positive"]:
        root = "./change-my-view-modes/v2.0/" + t + "/"
        for f in os.listdir(root):
            filename = os.path.join(root, f)

            if not (os.path.isfile(filename) and f.endswith(".xml")):
                continue

            for elem in generate_components(filename):
                yield elem


"""NOTES:
1. Deltas(&#8710) and some other symbols not parsed correctly without this double parsing. &gt; (>) , &#8217(apostrophe) get parsed fine with just using 'xml' and lxml.
2. &gt; doesn't get parsed to > when using str(post). But when doing post.get_text(), all the tags inside post will be removed and &gt; will be parsed to ">". 
   The &# characters in initial xml_string are all converted to proper unicode versions, the moment we parse with "lxml".
"""
